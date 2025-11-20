import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Optional

class SpeakerConsistencyLoss(nn.Module):
    """
    Computes similarity between the generated Style Vector and the Ground Truth Speaker Embedding.
    Uses Cosine Similarity.
    """
    def __init__(self, style_dim, spkr_dim, proj_dropout=0.1):
        super().__init__()
        self.style_dim = style_dim
        self.spkr_dim = spkr_dim
        
        # Projection if dimensions mismatch
        if style_dim != spkr_dim:
            self.proj = nn.Sequential(
                nn.Linear(style_dim, spkr_dim),
                nn.ReLU(),
                nn.Dropout(proj_dropout),
                nn.Linear(spkr_dim, spkr_dim)
            )
        else:
            self.proj = nn.Identity()
            
        self.loss_fn = nn.CosineEmbeddingLoss()

    def forward(self, style_vec, gt_spkr_emb):
        """
        style_vec: (B, style_dim) - from StyleEncoder
        gt_spkr_emb: (B, spkr_dim) - from PyAnnote/Dataset
        """
        # Project style vector to speaker space
        proj_style = self.proj(style_vec)
        
        # Normalize vectors (CosineEmbeddingLoss expects inputs, but pre-normalization helps stability)
        # Note: CosineEmbeddingLoss takes target=1 for similarity
        target = torch.ones(style_vec.size(0), device=style_vec.device)
        
        loss = self.loss_fn(proj_style, gt_spkr_emb, target)
        return loss

class MOSLoss(nn.Module):
    """
    Wrapper for UTMOS or similar MOS prediction models.
    Note: Optimizing MOS directly on AR logits is non-differentiable without Gumbel-Softmax
    and a differentiable vocoder. This module provides the computation mechanism,
    primarily for validation or RL-based fine-tuning.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.predictor = None
        # We load lazily to avoid overhead if not used
        
    def load_predictor(self):
        if self.predictor is None:
            try:
                # Load UTMOS from torchhub
                self.predictor = torch.hub.load("fwilhelm/wvmos", "wvmos", trust_repo=True).to(self.device)
                self.predictor.eval()
                for p in self.predictor.parameters():
                    p.requires_grad = False
            except Exception as e:
                print(f"[MOSLoss] Failed to load UTMOS predictor: {e}")
                self.predictor = None

    def forward(self, audio_wavs, sample_rate=16000):
        """
        audio_wavs: (B, T) or (B, 1, T) raw waveform
        Returns: (B,) MOS scores
        """
        if self.predictor is None:
            self.load_predictor()
        
        if self.predictor is None:
            return torch.zeros(audio_wavs.size(0), device=audio_wavs.device, requires_grad=True)
            
        if audio_wavs.dim() == 3:
            audio_wavs = audio_wavs.squeeze(1)
            
        # Resample if necessary (UTMOS usually expects 16k)
        # Assuming input is 16k for now as per config
        
        try:
            return self.predictor(audio_wavs)
        except Exception as e:
            print(f"[MOSLoss] Error during calculation: {e}")
            return torch.zeros(audio_wavs.size(0), device=audio_wavs.device, requires_grad=True)

class SISDRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) Loss.
    Commonly used for speech separation and enhancement evaluation.
    Negative SI-SDR is minimized.
    """
    def __init__(self, zero_mean=True, eps=1e-8):
        super().__init__()
        self.zero_mean = zero_mean
        self.eps = eps

    def forward(self, preds, target):
        """
        preds:  (B, T) or (B, 1, T) Estimate
        target: (B, T) or (B, 1, T) Reference
        """
        if preds.dim() == 3:
            preds = preds.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
            
        assert preds.shape == target.shape, f"Shape mismatch: {preds.shape} vs {target.shape}"
        
        if self.zero_mean:
            mean_preds = preds.mean(dim=-1, keepdim=True)
            mean_target = target.mean(dim=-1, keepdim=True)
            preds = preds - mean_preds
            target = target - mean_target
            
        # Calculate scalar projection (alpha)
        # alpha = <x, s> / <s, s>
        dot = torch.sum(preds * target, dim=-1, keepdim=True)
        s_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        alpha = dot / s_energy
        
        # Scaled target
        e_target = alpha * target
        
        # Noise component
        e_noise = preds - e_target
        
        # SI-SDR
        # 10 * log10( ||e_target||^2 / ||e_noise||^2 )
        target_energy = torch.sum(e_target ** 2, dim=-1) + self.eps
        noise_energy = torch.sum(e_noise ** 2, dim=-1) + self.eps
        
        si_sdr = 10 * torch.log10(target_energy / noise_energy)
        
        # Return negative mean SI-SDR for minimization
        return -si_sdr.mean()

