from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from trainer_style import StyleWav2TTS
from pytorch_lightning.strategies import DDPStrategy
import argparse
import json
import os
import math

def main():
    parser = argparse.ArgumentParser()

    #Paths
    parser.add_argument('--saving_path', type=str, default='./ckpt_style')
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--codec_model_id', type=str, default='hf-audio/xcodec-wavlm-more-data')
    parser.add_argument('--codec_bandwidth', type=float, default=2.0)
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--metapath', type=str, required=True)
    parser.add_argument('--val_metapath', type=str, required=True)
    parser.add_argument('--sampledir', type=str, default='./logs_style')
    parser.add_argument('--pretrained_path', type=str, default=None)
    # speaker_embedding_dir might be unused if we compute from mel, but keeping for compatibility
    parser.add_argument('--speaker_embedding_dir', type=str, default=None)

    #Optimizer
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=150)
    parser.add_argument('--train_bucket_size', type=int, default=8192)
    parser.add_argument('--optim_flat_percent', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=float, default=2.0)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)

    #Architecture
    parser.add_argument('--ffd_size', type=int, default=2048, help='Reduced from 3072 for ~100M total params')
    parser.add_argument('--hidden_size', type=int, default=512, help='Reduced from 768 for ~100M total params')
    parser.add_argument('--enc_nlayers', type=int, default=3, help='Reduced from 4')
    parser.add_argument('--dec_nlayers', type=int, default=3, help='Increased to 3 to balance, or keep 2')
    parser.add_argument('--nheads', type=int, default=8, help='Reduced from 12')
    parser.add_argument('--ar_layer', type=int, default=1)
    parser.add_argument('--ar_ffd_size', type=int, default=1024)
    parser.add_argument('--ar_hidden_size', type=int, default=256)
    parser.add_argument('--ar_nheads', type=int, default=4)
    parser.add_argument('--aligner_softmax_temp', type=float, default=1.0)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-5)
    
    #Vocoder Training
    parser.add_argument('--freeze_vocoder', action='store_true', default=False, help='Freeze vocoder weights (default: False, trainable)')
    parser.add_argument('--vocoder_lr', type=float, default=1e-5, help='Learning rate for vocoder fine-tuning')

    #Dropout
    parser.add_argument('--speaker_embed_dropout', type=float, default=0.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    #Trainer
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=2)
    parser.add_argument('--precision', type=str, choices=['16-mixed', '32-true', "bf16-mixed"], default='32-true')
    parser.add_argument('--nworkers', type=int, default=16)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--version', type=int, default=None)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--save_every_n_epochs', type=int, default=1)
    parser.add_argument('--enable_debug_logger', action='store_true', default=True,
                        help='Enable structured debug logging to file (default: True)')
    parser.add_argument('--debug_log_dir', type=str, default=None,
                        help='Directory for debug logs (default: logs_style/debug)')

    #Sampling
    parser.add_argument('--use_repetition_token', action='store_true')
    parser.add_argument('--use_repetition_gating', action='store_true')
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--sampling_temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--min_top_k', type=int, default=3)
    parser.add_argument('--top_p', type=float, default=0.7)
    parser.add_argument('--sample_num', type=int, default=4)
    parser.add_argument('--length_penalty_max_length', type=int, default=15000)
    parser.add_argument('--length_penalty_max_prob', type=float, default=0.95)
    parser.add_argument('--max_input_length', type=int, default=2048)
    parser.add_argument('--max_output_length', type=int, default=1500)
    parser.add_argument('--phone_context_window', type=int, default=3)
    
    #Losses
    parser.add_argument('--lambda_spk', type=float, default=1.0, help='Weight for Speaker Consistency Loss')
    parser.add_argument('--lambda_mos', type=float, default=0.0, help='Weight for MOS Loss (if applicable)')
    parser.add_argument('--lambda_sisdr', type=float, default=0.0, help='Weight for SI-SDR Loss (computationally expensive)')

    #Data
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--n_codes', type=int, default=512)
    parser.add_argument('--source_n_codes', type=int, default=1024,
                        help='Original codec vocabulary size before downsampling; '
                             'set higher than n_codes to downsample metadata on the fly.')
    parser.add_argument('--n_cluster_groups', type=int, default=4)

    args = parser.parse_args()

    # Ensure checkpoint directory exists
    os.makedirs(args.saving_path, exist_ok=True)
    print(f"[Setup] Checkpoint directory: {os.path.abspath(args.saving_path)}")
    
    with open(os.path.join(args.saving_path, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    fname_prefix = f''

    # Setup strategy for distributed training
    strategy = None
    if args.distributed:
        strategy = DDPStrategy(find_unused_parameters=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.saving_path,
        filename=(fname_prefix+'{epoch}-{step}'),
        every_n_epochs=args.save_every_n_epochs,
        verbose=True,
        save_last=True,
        save_on_train_epoch_end=True
    )
    
    print(f"[Setup] Checkpoints will be saved every {args.save_every_n_epochs} epoch(s) to: {args.saving_path}")

    logger = TensorBoardLogger(args.sampledir, name="VQ-TTS-Style", version=args.version)

    wrapper = Trainer(
        precision=args.precision,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        max_epochs=args.max_epochs,
        devices=(-1 if args.distributed else 1),
        accelerator=args.accelerator,
        use_distributed_sampler=False,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=logger,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        strategy=strategy,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm='norm'
    )
    model = StyleWav2TTS(args)
    
    # Log Setup info
    train_samples = len(model.data)
    approx_steps_per_epoch = math.ceil(train_samples / max(1, args.batch_size))
    eff_steps_per_epoch = math.ceil(approx_steps_per_epoch / max(1, args.accumulate_grad_batches))
    est_total_steps = eff_steps_per_epoch * max(1, args.max_epochs)
    print(f"[Setup] Training examples: {train_samples}")
    print(f"[Setup] Max epochs: {args.max_epochs}, approx. steps/epoch: {approx_steps_per_epoch}")
    print(f"[Setup] Grad accumulation: {args.accumulate_grad_batches}, optimizer steps/epoch: {eff_steps_per_epoch}, est. total optimizer steps: {est_total_steps}")
    print(f"[Setup] Validation every {args.check_val_every_n_epoch} epoch(s); checkpoints every {args.save_every_n_epochs} epoch(s).")
    lightning_est = getattr(wrapper, "estimated_stepping_batches", None)
    if lightning_est is not None:
        print(f"[Setup] Lightning estimated stepping batches: {lightning_est}")
        
    wrapper.fit(model, ckpt_path=args.resume_checkpoint)


if __name__ == "__main__":
    main()

