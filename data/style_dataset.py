import os
from torch.utils import data
import torch
import json
import numpy as np
import soundfile as sf
import random
from librosa.util import normalize
from pyannote.audio import Inference
import librosa
from .QuantizeDataset import QuantizeDataset

class StyleQuantizeDataset(QuantizeDataset):
    def __init__(self, hp, metapath):
        super().__init__(hp, metapath)
        
    def get_mel(self, audio):
        # Calculate Mel Spectrogram
        # Assume 16kHz, hop 320 (20ms), win 1024 (64ms), 80 mels
        # Adjust parameters as needed to match StyleTTS2 or your needs
        mel = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.hp.sample_rate, 
            n_fft=1024, 
            hop_length=320, 
            win_length=1024, 
            n_mels=80,
            fmin=0, 
            fmax=8000
        )
        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
        return mel.T # (T, n_mels)

    def __getitem__(self, i):
        speaker_embedding, quantization_s, quantization_e, phonemes, dataname = super().__getitem__(i)
        
        # Load audio again for Mel calculation (optimize if needed)
        audio = self._load_audio(dataname)
        mel = self.get_mel(audio)
        
        return (
            speaker_embedding, 
            quantization_s, 
            quantization_e, 
            phonemes, 
            dataname,
            torch.FloatTensor(mel)
        )

    def seqCollate(self, batch):
        # Extract standard batch items
        base_batch = [b[:4] + (b[4],) for b in batch] # exclude mel
        output = super().seqCollate(base_batch)
        
        # Handle Mel
        mels = [b[5] for b in batch]
        max_mel_len = max([m.size(0) for m in mels])
        
        # Pad Mels
        padded_mels = []
        mel_masks = []
        for mel in mels:
            pad_len = max_mel_len - mel.size(0)
            padded_mel = torch.nn.functional.pad(mel, (0, 0, 0, pad_len))
            padded_mels.append(padded_mel)
            
            mask = torch.zeros(max_mel_len, dtype=torch.bool)
            mask[mel.size(0):] = True
            mel_masks.append(mask)
            
        output['mel'] = torch.stack(padded_mels) # (B, T, 80)
        output['mel'] = output['mel'].permute(0, 2, 1) # (B, 80, T) for StyleEncoder
        output['mel_mask'] = torch.stack(mel_masks)
        
        return output

class StyleQuantizeDatasetVal(StyleQuantizeDataset):
    def __getitem__(self, i):
        # Return structure must match val_dataloader expectation
        # For val, we usually need ground truth audio too
        speaker_embedding, quantization_s, quantization_e, phonemes, dataname = QuantizeDataset.__getitem__(self, i) # Call grandparent
        audio = self._load_audio(dataname)
        mel = self.get_mel(audio)
        
        return (
            torch.FloatTensor(speaker_embedding),
            torch.LongTensor(quantization_s),
            torch.LongTensor(quantization_e),
            torch.LongTensor(phonemes),
            torch.FloatTensor(audio),
            torch.FloatTensor(mel)
        )

