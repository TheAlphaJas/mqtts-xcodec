import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data.style_dataset import StyleQuantizeDataset, StyleQuantizeDatasetVal
from data.sampler import RandomBucketSampler
from modules.styletts2_transformer import StyleTTSDecoder
from modules.transformers import TransformerEncoderLayer, TransformerEncoder, TransformerDecoder, TransformerDecoderLayer
from modules.vocoder import Vocoder
from modules.losses import SpeakerConsistencyLoss, MOSLoss, SISDRLoss
from torch.utils import data
import pytorch_lightning.core.module as pl
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
plt.switch_backend('agg')

class DebugLogger:
    """Structured logger for training diagnostics"""
    def __init__(self, log_dir, enabled=True):
        self.enabled = enabled
        if not self.enabled:
            return
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"debug_log_{int(time.time())}.jsonl"
        self.summary_file = self.log_dir / f"debug_summary_{int(time.time())}.json"
        self.buffer = []
        self.buffer_size = 100
        self.start_time = time.time()
        
    def log(self, event_type, step, epoch, data):
        """Log an event with structured data"""
        if not self.enabled:
            return
        entry = {
            "timestamp": time.time() - self.start_time,
            "event_type": event_type,
            "step": step,
            "epoch": epoch,
            "data": data
        }
        self.buffer.append(entry)
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Write buffer to disk"""
        if not self.enabled or not self.buffer:
            return
        with open(self.log_file, 'a') as f:
            for entry in self.buffer:
                f.write(json.dumps(entry) + '\n')
        self.buffer = []
    
    def write_summary(self, summary_data):
        """Write final summary"""
        if not self.enabled:
            return
        self.flush()
        with open(self.summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

class StyleWav2TTS(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self._cached_total_steps = None
        self._optimizer_state_log = {}
        self._param_snapshot = {}
        
        # Initialize debug logger
        debug_log_dir = getattr(hp, 'debug_log_dir', os.path.join(getattr(hp, 'sampledir', './logs'), 'debug'))
        self.debug_logger = DebugLogger(debug_log_dir, enabled=getattr(hp, 'enable_debug_logger', True))
        
        self.data = StyleQuantizeDataset(hp, hp.metapath)
        self.val_data = StyleQuantizeDatasetVal(hp, hp.val_metapath)
        self.TTSdecoder = StyleTTSDecoder(hp, len(self.data.phoneset))
        self.n_decode_codes = self.TTSdecoder.transducer.n_decoder_codes
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=self.hp.label_smoothing)
        
        # Auxiliary Losses
        # Assuming gt speaker embedding dim is 512 (from pyannote)
        # Style vector dim is hidden_size (e.g. 768)
        self.spk_loss = SpeakerConsistencyLoss(style_dim=hp.hidden_size, spkr_dim=512)
        self.mos_loss = MOSLoss()
        self.sisdr_loss = SISDRLoss()
        
        self.phone_embedding = nn.Embedding(len(self.data.phoneset), hp.hidden_size, padding_idx=self.data.phoneset.index('<pad>'))
        
        if self.hp.pretrained_path:
            self.load()
        else:
            self.apply(self.init_weights)
        self.vocoder = Vocoder(
            hp.codec_model_id,
            hp.codec_bandwidth,
            sample_rate=hp.sample_rate,
            codebook_limit=hp.n_codes
        )
        
        if getattr(hp, 'freeze_vocoder', True):
            self.vocoder.eval()
            for param in self.vocoder.parameters():
                param.requires_grad = False
            self.print("[Model] Vocoder is FROZEN.")
        else:
            self.vocoder.train()
            for param in self.vocoder.parameters():
                param.requires_grad = True
            self.print("[Model] Vocoder is TRAINABLE (fine-tuning enabled).")

    def load(self):
        state_dict = torch.load(self.hp.pretrained_path)['state_dict']
        self.load_state_dict(state_dict, strict=False)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            # PyTorch 2.1: Manually zero out padding index if it exists
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight.data[module.padding_idx].fill_(0)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def train_dataloader(self):
        length = self.data.lengths
        # PyTorch Lightning 2.x: world_size and local_rank are still available
        # Use num_devices * num_nodes for world_size, and global_rank for rank
        if self.hp.distributed and self.trainer is not None:
            world_size = getattr(self.trainer, 'world_size', getattr(self.trainer, 'num_devices', 1) * getattr(self.trainer, 'num_nodes', 1))
            rank = getattr(self.trainer, 'local_rank', getattr(self.trainer, 'global_rank', 0))
        else:
            world_size = 1
            rank = 0
        sampler = RandomBucketSampler(self.hp.train_bucket_size, length, self.hp.batch_size, drop_last=True, distributed=self.hp.distributed,
                                      world_size=world_size, rank=rank)
        dataset = data.DataLoader(self.data,
                                  num_workers=self.hp.nworkers,
                                  batch_sampler=sampler,
                                  collate_fn=self.data.seqCollate)
        return dataset

    def val_dataloader(self):
        dataset = data.DataLoader(self.val_data,
                                  num_workers=self.hp.nworkers,
                                  shuffle=False)
        return dataset

    def configure_optimizers(self):
        # Separate parameter groups for different learning rates
        vocoder_params = []
        main_params = []
        
        if not getattr(self.hp, 'freeze_vocoder', True):
            vocoder_params = list(self.vocoder.parameters())
            # Identify vocoder params by object identity
            vocoder_param_ids = set(id(p) for p in vocoder_params)
            main_params = [p for p in self.parameters() if id(p) not in vocoder_param_ids]
        else:
            main_params = list(self.parameters())
            
        param_groups = [
            {'params': main_params, 'lr': self.hp.lr, 'betas': (self.hp.adam_beta1, self.hp.adam_beta2)}
        ]
        
        if vocoder_params:
            voc_lr = getattr(self.hp, 'vocoder_lr', 1e-5)
            param_groups.append(
                {'params': vocoder_params, 'lr': voc_lr, 'betas': (self.hp.adam_beta1, self.hp.adam_beta2)}
            )
            
        optimizer_adam = optim.Adam(param_groups)

        def lambda_lr(current_step: int):
            total_steps, num_warmup_steps, num_flat_steps = self._scheduler_factors()
            # Warmup phase: linear ramp from 0 to 1
            if num_warmup_steps > 0 and current_step < num_warmup_steps:
                # Start from small value, not 0
                return max(0.01, float(current_step + 1) / float(max(1, num_warmup_steps)))
            # Flat/plateau phase: stay at 1.0
            if num_warmup_steps <= current_step < (num_warmup_steps + num_flat_steps):
                return 1.0
            # Decay phase: linear decay
            decay_steps = total_steps - (num_warmup_steps + num_flat_steps)
            if decay_steps <= 0:
                return 1.0  # If no decay steps, stay at 1.0
            remaining = max(0, total_steps - current_step)
            return max(0.1, float(remaining) / float(decay_steps))  # Don't go below 0.1

        scheduler_adam = {
            'scheduler': optim.lr_scheduler.LambdaLR(optimizer_adam, lambda_lr),
            'interval': 'step'
        }
        return [optimizer_adam], [scheduler_adam]

    def _scheduler_factors(self):
        total_steps = self._resolve_total_steps()
        steps_per_epoch = max(1, total_steps // max(1, self.hp.max_epochs))
        if self.hp.warmup_epochs > 0:
            num_warmup_steps = int(self.hp.warmup_epochs * steps_per_epoch)
            num_warmup_steps = max(1, min(total_steps, num_warmup_steps))
        else:
            num_warmup_steps = 0
        num_flat_steps = int(self.hp.optim_flat_percent * total_steps)
        num_flat_steps = max(0, min(total_steps, num_flat_steps))
        return total_steps, num_warmup_steps, num_flat_steps

    def _resolve_total_steps(self):
        if self._cached_total_steps is not None:
            return self._cached_total_steps
        trainer = getattr(self, "trainer", None)
        total_steps = None
        if trainer is not None:
            total_steps = getattr(trainer, "estimated_stepping_batches", None)
            if total_steps is None:
                num_batches = getattr(trainer, "num_training_batches", None)
                if num_batches not in (None, float("inf")):
                    accum = getattr(trainer, "accumulate_grad_batches", getattr(self.hp, 'accumulate_grad_batches', 1))
                    epochs = trainer.max_epochs or self.hp.max_epochs or 1
                    total_steps = math.ceil(num_batches / max(1, accum)) * max(1, epochs)
        if total_steps is None:
            dataset_size = len(self.data)
            approx_batch = max(1, getattr(self.hp, 'batch_size', 1))
            steps_per_epoch = math.ceil(dataset_size / approx_batch)
            total_steps = max(1, steps_per_epoch * max(1, self.hp.max_epochs))
        self._cached_total_steps = max(1, int(total_steps))
        return self._cached_total_steps

    def on_fit_start(self):
        if self.trainer is None:
            return
        self._cached_total_steps = None
        
        # Check trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = total_params - trainable_params
        
        # List frozen modules
        frozen_modules = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                frozen_modules.append(name)
        
        total_examples = len(self.data)
        approx_steps_per_epoch = math.ceil(total_examples / max(1, self.hp.batch_size))
        eff_steps_per_epoch = math.ceil(approx_steps_per_epoch / max(1, self.hp.accumulate_grad_batches))
        est_total_steps = eff_steps_per_epoch * max(1, self.hp.max_epochs)
        lightning_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        resolved_total = self._resolve_total_steps()
        
        # Log to debug file
        self.debug_logger.log("fit_start", 0, 0, {
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "total_params": total_params,
            "frozen_modules": frozen_modules,
            "total_examples": total_examples,
            "max_epochs": self.hp.max_epochs,
            "approx_steps_per_epoch": approx_steps_per_epoch,
            "optimizer_steps_per_epoch": eff_steps_per_epoch,
            "est_total_optimizer_steps": est_total_steps,
            "lightning_estimated_steps": lightning_steps,
            "resolved_total_steps": resolved_total,
            "check_val_every_n_epoch": self.hp.check_val_every_n_epoch,
            "save_every_n_epochs": self.hp.save_every_n_epochs,
        })
        
        # Minimal terminal output
        self.print(f"[Setup] Training: {trainable_params:,} params, {total_examples} examples, "
                  f"{self.hp.max_epochs} epochs, ~{eff_steps_per_epoch} steps/epoch")
        self.print(f"[Setup] Debug logs: {self.debug_logger.log_file}")
        
        self._log_optimizer_state(context="fit_start")

    def on_train_epoch_start(self):
        # Store parameter snapshot for gradient tracking
        for name, param in self.named_parameters():
            if param.requires_grad:
                self._param_snapshot[name] = param.data.clone()
        
        # Log optimizer state at epoch start
        opt_state = self._get_optimizer_state_dict()
        self.debug_logger.log("epoch_start", self.global_step, self.current_epoch, {
            "optimizer_state": opt_state
        })
        self.print(f"[Epoch {self.current_epoch}] Starting (step {self.global_step})")
        self._log_optimizer_state(context=f"epoch_{self.current_epoch}")
    
    def on_before_optimizer_step(self, optimizer):
        # Collect gradient statistics
        if self.global_step % 10 == 0:  # Every 10 steps
            total_norm = 0.0
            max_grad = 0.0
            min_grad = float('inf')
            grad_norms = {}
            num_params_with_grad = 0
            num_params_total = 0
            for name, param in self.named_parameters():
                if param.requires_grad:
                    num_params_total += 1
                    if param.grad is not None:
                        num_params_with_grad += 1
                        param_norm = param.grad.data.norm(2).item()
                        total_norm += param_norm ** 2
                        max_grad = max(max_grad, param.grad.abs().max().item())
                        min_grad = min(min_grad, param.grad.abs().min().item())
                        if param_norm > 0.1:  # Only log significant gradients
                            grad_norms[name] = param_norm
            total_norm = total_norm ** 0.5
            
            top_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:10]
            
            self.debug_logger.log("gradient_stats", self.global_step, self.current_epoch, {
                "total_norm": total_norm,
                "max_grad": max_grad,
                "min_grad": min_grad,
                "num_params_with_grad": num_params_with_grad,
                "num_params_total": num_params_total,
                "top_10_grad_norms": {name: norm for name, norm in top_grads}
            })
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Track parameter updates every 50 steps
        if self.global_step % 50 == 0 and len(self._param_snapshot) > 0:
            param_changes = {}
            for name, param in self.named_parameters():
                if param.requires_grad and name in self._param_snapshot:
                    old_param = self._param_snapshot[name]
                    change = (param.data - old_param).abs().mean().item()
                    if change > 1e-8:
                        param_changes[name] = change
                    self._param_snapshot[name] = param.data.clone()
            
            top_changes = sorted(param_changes.items(), key=lambda x: x[1], reverse=True)[:10]
            
            self.debug_logger.log("param_updates", self.global_step, self.current_epoch, {
                "num_params_updated": len(param_changes),
                "top_10_param_changes": {name: change for name, change in top_changes},
                "no_updates": len(param_changes) == 0
            })
            
            if len(param_changes) == 0:
                self.print(f"[WARNING Step {self.global_step}] NO PARAMETER UPDATES!")

    def training_step(self, batch, batch_idx):
        # Note: batch now has 'mel' as output['mel']
        
        #Deal with speaker embedding (Using Style Encoder)
        # batch['speaker'] is the original speaker embedding (unused if relying on style encoder)
        # batch['mel'] is used for style encoding
        
        # Pass mel to decoder (it will handle style encoding)
        mel = batch['mel']
        gt_spkr_emb = batch['speaker']
        
        #Deal with phone segments
        phone_features = self.phone_embedding(batch['phone'])
        
        #Run decoder
        # Note: TTSDecoder.forward signature updated to accept mel
        recons_segments = self.TTSdecoder(
            batch['tts_quantize_input'], 
            phone_features, 
            batch['speaker'], # Passed but likely ignored/overwritten in StyleTTSDecoder
            mel,
            batch['quantize_mask'], 
            batch['phone_mask']
        )
        
        target = recons_segments['logits'][~batch['quantize_mask']].view(-1, self.n_decode_codes)
        labels = batch['tts_quantize_output'][~batch['quantize_mask']].view(-1)
        
        # Sanity check: ensure labels are in valid range (first step only)
        if self.global_step == 0:
            label_min = labels.min().item()
            label_max = labels.max().item()
            self.debug_logger.log("data_sanity_check", self.global_step, self.current_epoch, {
                "label_min": label_min,
                "label_max": label_max,
                "expected_max": self.n_decode_codes - 1,
                "labels_valid": 0 <= label_min and label_max < self.n_decode_codes
            })
            if label_max >= self.n_decode_codes or label_min < 0:
                self.print(f"[ERROR] Labels out of range: [{label_min}, {label_max}], expected [0, {self.n_decode_codes-1}]")
        
        # 1. Cross Entropy Loss (Main AR objective)
        loss_ce = self.cross_entropy(target, labels)
        acc = (target.argmax(-1) == labels).float().mean()
        
        # 2. Speaker Consistency Loss (Style Encoder objective)
        # Retrieve style vector from decoder output
        style_vec = recons_segments['style_vector']
        loss_spk = self.spk_loss(style_vec, gt_spkr_emb)
        
        # 3. SI-SDR Loss (if using differentiable vocoder approximation or if we just want to monitor)
        # Since vocoder is fixed and discrete, we can't backprop SI-SDR easily through the AR steps.
        # But we can compute it for monitoring or if we add a differentiable path later.
        # For now, calculating SI-SDR between GT audio and Vocoded GT audio (reconstruction upper bound)
        # OR between GT audio and Vocoded Predicted Codes (inference approximation).
        # Calculating fully differentiable SI-SDR requires Gumbel-Softmax codes -> Differentiable Vocoder.
        # Assuming we just want to log it for now or optimize style encoder if applicable.
        
        # Let's calculate it on the Style Encoder's ability to reconstruct the style? 
        # No, SI-SDR is waveform level.
        
        # Implementation:
        # We can't cheaply generate waveform for the whole batch every step (expensive).
        # So we'll compute SI-SDR only for logging or if weight > 0 and we accept the cost.
        # Given the current architecture (discrete codes), we skip backprop for SI-SDR on the AR model.
        # However, if we want to optimize the Style Encoder, we could potentially decode the GT codes with the predicted Style Vector?
        # Let's try: Reconstruct GT audio using GT codes + Predicted Style Vector. 
        # Compare this against GT audio using GT codes + GT Speaker Embedding (or just GT audio).
        # This optimizes the Style Encoder to produce a style vector that makes the vocoder happy.
        
        loss_sisdr = torch.tensor(0.0, device=loss_ce.device)
        lambda_sisdr = getattr(self.hp, 'lambda_sisdr', 0.0)
        
        if lambda_sisdr > 0 or (batch_idx % 50 == 0): # Calculate if weighted or for logging
             # Reconstruct using GT codes (q_s) and Predicted Style Vector
             # q_s is (B, T) indices. Vocoder needs indices.
             # We use q_s[:, 1:] (remove start token)
             codes_gt = batch['tts_quantize_input'][:, 1:]
             
             # Generate waveform (expensive!) - do it for a subset
             sub_size = min(2, codes_gt.size(0))
             codes_sub = codes_gt[:sub_size]
             style_sub = style_vec[:sub_size]
             gt_audio_sub = batch['audio'][:sub_size] if 'audio' in batch else None 
             
             if gt_audio_sub is not None:
                 # Decode with vocoder
                 # Note: Vocoder expects (B, n_quant, T_codes)
                 # codes_sub is (B, T_codes). Need to expand to (B, n_quant, T_codes)?
                 # Or is it (B, T_codes, n_quant)? 
                 # Dataset seqCollate: qs is padded with constant_values=n_codes
                 # qs is (B, T). It seems we are using 1 code per step here?
                 # MQTTS usually uses multiple codes per step if n_cluster_groups > 1
                 # Let's check data pipeline. 
                 # QuantizeDataset: quantization is (T, 4). 
                 # seqCollate: qs = np.pad(qs, [[0, max_len], [0,0]]) -> (B, T, 4)
                 # BUT batch['tts_quantize_input'] is LongTensor(output[k]).
                 # Wait, if qs is (B, T, 4), then codes_gt is (B, T, 4).
                 
                 # The vocoder wrapper expects: codes (B, T, n_q) or (B, n_q, T)?
                 # Vocoder.forward: if dim==3 -> (B, T, groups). 
                 # So codes_sub should be (B, T, 4).
                 
                 try:
                     # Normalize style vector for vocoder if needed
                     norm_style_sub = F.normalize(style_sub, dim=-1)
                     
                     rec_audio = self.vocoder(codes_sub, norm_style_sub) # (B, 1, T_wav)
                     rec_audio = rec_audio.squeeze(1) # (B, T_wav)
                     
                     # Align lengths
                     min_len = min(rec_audio.size(1), gt_audio_sub.size(1))
                     rec_audio = rec_audio[:, :min_len]
                     gt_audio_sub = gt_audio_sub[:, :min_len]
                     
                     # Calculate SI-SDR
                     val_sisdr = self.sisdr_loss(rec_audio, gt_audio_sub)
                     
                     if lambda_sisdr > 0:
                         loss_sisdr = val_sisdr
                         
                     # Log it
                     self.log("train/sisdr", -val_sisdr, on_step=True, logger=True) # Log positive SI-SDR (higher is better)
                     
                 except Exception as e:
                     print(f"SI-SDR Calc Error: {e}")

        # Combine losses
        lambda_spk = getattr(self.hp, 'lambda_spk', 1.0)
        
        total_loss = loss_ce + lambda_spk * loss_spk + lambda_sisdr * loss_sisdr
        
        # Log training metrics every 10 steps
        if self.global_step % 10 == 0:
            opt = self.optimizers()
            lr = opt.param_groups[0]['lr'] if hasattr(opt, 'param_groups') else 0.0
            
            # Sample predictions vs labels
            sample_size = min(10, len(labels))
            pred_samples = target[:sample_size].argmax(-1).tolist()
            label_samples = labels[:sample_size].tolist()
            
            self.debug_logger.log("training_step", self.global_step, self.current_epoch, {
                "loss_total": total_loss.item(),
                "loss_ce": loss_ce.item(),
                "loss_spk": loss_spk.item(),
                "accuracy": acc.item(),
                "learning_rate": lr,
                "target_shape": list(target.shape),
                "labels_shape": list(labels.shape),
                "num_valid_tokens": (~batch['quantize_mask']).sum().item(),
                "sample_predictions": pred_samples,
                "sample_labels": label_samples,
                "batch_idx": batch_idx
            })
        
        # Minimal terminal output every 100 steps
        if self.global_step % 100 == 0:
            opt = self.optimizers()
            lr = opt.param_groups[0]['lr'] if hasattr(opt, 'param_groups') else 0.0
            self.print(f"[Step {self.global_step}] Loss: {total_loss.item():.4f} (CE: {loss_ce.item():.4f}, Spk: {loss_spk.item():.4f}), Acc: {acc.item():.4f}, LR: {lr:.2e}")
        
        self.log("train/loss", total_loss, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss_ce", loss_ce, on_step=True, logger=True)
        self.log("train/loss_spk", loss_spk, on_step=True, logger=True)
        self.log("train/acc", acc, on_step=True, prog_bar=True, logger=True)
        return total_loss

    def on_validation_epoch_start(self):
        #For the first half samples, and random choose the rest half
        start_point, half = 4, self.hp.sample_num // 2
        if self.hp.sample_num > 0:
            self.sample_idxs = list(range(start_point, start_point + half)) + \
                np.random.randint(low=start_point + half, high=len(self.val_data), size=self.hp.sample_num//2).tolist()
        else:
            self.sample_idxs = []

    def validation_step(self, batch, batch_idx):
        #Batch size = 1
        spkr, q_s, q_e, phone, ground_truth, mel = batch # Unpack mel
        
        norm_spkr = F.normalize(spkr, dim=-1)
        # spkr = self.spkr_linear(norm_spkr) # Removed/Replaced by Style Encoder
        
        phone_features = self.phone_embedding(phone)
        
        # Pass mel
        recons_segments = self.TTSdecoder(q_s, phone_features, norm_spkr, mel, None, None)
        
        target = recons_segments['logits'].view(-1, self.n_decode_codes)
        labels = q_e.view(-1)
        loss = self.cross_entropy(target, labels)
        acc = (target.argmax(-1) == labels).float().mean()
        self.log("val/loss", loss, on_epoch=True, logger=True)
        self.log("val/acc", acc, on_epoch=True, logger=True)

        #Run inference with bs = 1
        if batch_idx in self.sample_idxs:
            batch_idx = self.sample_idxs.index(batch_idx)
            phone_mask = torch.full((phone_features.size(0), phone_features.size(1)), False, dtype=torch.bool, device=phone_features.device)
            
            # Inference with Style Encoder
            synthetic, infer_attn = self.TTSdecoder.inference_topkp_sampling_batch(
                phone_features, mel, phone_mask, prior=None, output_alignment=True
            )
            
            synthetic = synthetic[0].unsqueeze(0)
            
            # Note: Vocoder also needs style vector? 
            # If vocoder is WaveLM based, it usually takes the discrete codes.
            # It might also take a speaker embedding.
            # For now, we pass the computed style vector as speaker embedding if the vocoder supports it.
            # But the existing vocoder wrapper takes `speaker_embedding` in forward.
            # `TTSDecoder` computes style vector internally. We should expose it or recompute it.
            
            # Let's recompute style vector for vocoder conditioning
            style_vec = self.TTSdecoder.style_encoder(mel)
            style_vec = F.normalize(style_vec, dim=-1) # Normalize for vocoder? Or keep raw?
            # MQTTS usually used normalized spkr embedding. Let's try raw or normalized.
            # Assuming Vocoder expects similar stats to original spkr embedding if it was trained with one.
            
            synthetic = self.vocoder(synthetic, style_vec).float()
            
            #Reconstructed Audio with vocoder
            reconstructed_gt = self.vocoder(q_s[:, 1:], style_vec).float()
            
            synthetic = torch.clamp(synthetic, -1.0, 1.0)
            reconstructed_gt = torch.clamp(reconstructed_gt, -1.0, 1.0)
            
            #Write files
            sw = self.logger.experiment
            sw.add_audio(f'generated/{batch_idx}', synthetic, self.global_step, self.hp.sample_rate)
            sw.add_audio(f'vocoder-reconstructed/{batch_idx}', reconstructed_gt, self.global_step, self.hp.sample_rate)
            sw.add_audio(f'groundtruth/{batch_idx}', ground_truth[0], self.global_step, self.hp.sample_rate)

            #Plot attentions
            self.plot_attn(recons_segments['encoder_attention'], f'enc-attn/{batch_idx}', (10, 10))
            self.plot_attn(recons_segments['decoder_attention'], f'dec-attn/{batch_idx}', (10, 10))
            self.plot_attn([recons_segments['alignment']], f'train-alignment/{batch_idx}', (10, 10))
            self.plot_attn([infer_attn.unsqueeze(0)], f'infer-alignment/{batch_idx}', (10, 10))

    def plot_attn(self, attns, prefix, figsize):
        nheads = attns[0].size(1)
        fig, axs = plt.subplots(len(attns), nheads, constrained_layout=True, figsize=figsize)
        if len(attns) == 1 and nheads == 1:
            axs = [[axs]]
        elif len(attns) == 1 or nheads == 1:
            axs = [axs]
        for i, attn in enumerate(attns): #Each layers
            attn = attn.float().cpu().numpy()
            for j, head_attn in enumerate(attn[0]):
                axs[i][j].matshow(head_attn, aspect="auto", origin="lower", interpolation='none')
                if i != 0 or j != 0:
                    axs[i][j].axis('off')
        self.logger.experiment.add_figure(prefix, fig, self.global_step)
        plt.close()

    def _log_optimizer_state(self, context: str):
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        optimizers = getattr(trainer, "optimizers", None)
        if not optimizers:
            return
        for opt_idx, optimizer in enumerate(optimizers):
            for group_idx, group in enumerate(optimizer.param_groups):
                lr = float(group.get('lr', 0.0))
                betas = group.get('betas')
                wd = float(group.get('weight_decay', 0.0))
                prev = self._optimizer_state_log.get((opt_idx, group_idx))
                delta = None if prev is None else lr - prev
                delta_str = "n/a" if delta is None else f"{delta:+.3e}"
                beta_str = ""
                if isinstance(betas, (tuple, list)) and len(betas) == 2:
                    beta_str = f", betas=({betas[0]:.4f}, {betas[1]:.4f})"
                self.print(
                    f"[Optimizer] {context} opt{opt_idx}/group{group_idx}: lr={lr:.6e} (Δ {delta_str}){beta_str}, weight_decay={wd:.3g}"
                )
                self._optimizer_state_log[(opt_idx, group_idx)] = lr

    def _get_optimizer_state_dict(self):
        """Extract optimizer state as a dictionary"""
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return {}
        optimizers = getattr(trainer, "optimizers", None)
        if not optimizers:
            return {}
        
        result = []
        for opt_idx, optimizer in enumerate(optimizers):
            for group_idx, group in enumerate(optimizer.param_groups):
                result.append({
                    "optimizer_idx": opt_idx,
                    "group_idx": group_idx,
                    "lr": float(group.get('lr', 0.0)),
                    "betas": group.get('betas'),
                    "weight_decay": float(group.get('weight_decay', 0.0))
                })
        return result

    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        # Log epoch completion
        self.print(f"[Epoch {self.current_epoch}] Complete (step {self.global_step})")
        
        # Check if checkpoint should be saved this epoch
        if hasattr(self.hp, 'save_every_n_epochs') and self.hp.save_every_n_epochs > 0:
            if (self.current_epoch % self.hp.save_every_n_epochs) == 0:
                self.print(f"[Checkpoint] Saving checkpoint for epoch {self.current_epoch}...")

    def on_fit_end(self):
        """Flush debug logs and write summary at end of training"""
        self.debug_logger.flush()
        self.print(f"[Training Complete] Debug logs written to: {self.debug_logger.log_file}")

