import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, XcodecModel


class Vocoder(nn.Module):
    """
    Thin wrapper around Hugging Face's X-Codec model so it can be used anywhere the legacy
    quantizer/vocoder combo was previously expected.
    """

    def __init__(self, model_id: str = "hf-audio/xcodec-wavlm-more-data", bandwidth: float = 2.0,
                 sample_rate: int = 16000):
        super().__init__()
        self.model_id = model_id
        self.bandwidth = bandwidth
        self.model = XcodecModel.from_pretrained(model_id)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.sample_rate = sample_rate or getattr(self.feature_extractor, "sampling_rate", 16000)
        self.codebook_size = getattr(self.model.config, "codebook_size", 1024)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _prepare_audio_batch(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu()
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            elif audio.dim() != 2:
                raise ValueError("Audio tensor must have shape (samples) or (batch, samples).")
            batch = [a.numpy() for a in audio]
        elif isinstance(audio, np.ndarray):
            if audio.ndim == 1:
                batch = [audio]
            elif audio.ndim == 2:
                batch = [arr for arr in audio]
            else:
                raise ValueError("NumPy audio input must be 1D or 2D.")
        elif isinstance(audio, list):
            batch = [np.asarray(a) for a in audio]
        else:
            raise TypeError("Unsupported audio container type for encode().")
        processed = []
        for arr in batch:
            arr = np.asarray(arr, dtype=np.float32).reshape(-1)
            processed.append(arr)
        return processed

    def _trim_special_tokens(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Remove start/end/padding tokens that were appended outside of the codec domain.
        """
        if codes.dim() != 2:
            raise ValueError("Expect codes with shape (time, num_quantizers).")
        mask = torch.any(codes >= self.codebook_size, dim=-1)
        if mask.any():
            first_invalid = torch.nonzero(mask, as_tuple=False)[0, 0]
            codes = codes[:first_invalid]
        if codes.numel() == 0:
            raise ValueError("Sequence does not contain any valid codec tokens.")
        return codes

    @torch.no_grad()
    def forward(self, codes, speaker_embedding=None, return_lengths: bool = False):
        """
        Decode codec tokens back to waveform.
        """
        if isinstance(codes, torch.Tensor):
            if codes.dim() == 2:
                codes = codes.unsqueeze(0)
            elif codes.dim() != 3:
                raise ValueError("Codes tensor must have shape (batch, time, groups).")
            batch = [sample for sample in codes]
        elif isinstance(codes, list):
            batch = [torch.as_tensor(sample) for sample in codes]
        else:
            raise TypeError("Unsupported codes container type.")

        decoded_audio = []
        lengths = []
        for sample in batch:
            sample = sample.to(self.device).long()
            sample = self._trim_special_tokens(sample)
            sample = sample.transpose(0, 1).unsqueeze(0)  # 1, num_quantizers, length
            audio = self.model.decode(sample).audio_values.squeeze(0)
            decoded_audio.append(audio)
            lengths.append(audio.size(-1))

        max_len = max(lengths)
        padded = []
        for audio in decoded_audio:
            if audio.size(-1) < max_len:
                audio = F.pad(audio, (0, max_len - audio.size(-1)))
            padded.append(audio)
        batch_audio = torch.stack(padded, dim=0)
        batch_audio = torch.clamp(batch_audio, -1.0, 1.0)
        if return_lengths:
            return batch_audio, torch.tensor(lengths, device=batch_audio.device)
        return batch_audio

    @torch.no_grad()
    def encode(self, audio):
        """
        Encode raw audio waveforms into discrete codec tokens.
        """
        batch = self._prepare_audio_batch(audio)
        inputs = self.feature_extractor(
            raw_audio=batch,
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt"
        )
        input_values = inputs["input_values"].to(self.device)
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)
        codes = self.model.encode(input_values, bandwidth=self.bandwidth).audio_codes
        return codes.transpose(1, 2).long()
