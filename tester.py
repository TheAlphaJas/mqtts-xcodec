import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.wildttstransformer import TTSDecoder
from modules.transformers import TransformerEncoderLayer, TransformerEncoder, TransformerDecoder, TransformerDecoderLayer
from torch.utils import data
from modules.vocoder import Vocoder
import soundfile as sf
from librosa.util import normalize
from pyannote.audio import Inference
import random
from tqdm import tqdm

class Wav2TTS_infer(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.hp.init = 'std'
        self.TTSdecoder = TTSDecoder(hp, len(self.hp.phoneset))
        self.spkr_linear = nn.Linear(512, hp.hidden_size)
        self.phone_embedding = nn.Embedding(len(self.hp.phoneset), hp.hidden_size, padding_idx=self.hp.phoneset.index('<pad>'))
        self.load()
        self.spkr_embedding = Inference("pyannote/embedding", window="whole")
        self.vocoder = Vocoder(hp.codec_model_id, hp.codec_bandwidth, sample_rate=hp.sample_rate)

    def load(self):
        state_dict = torch.load(self.hp.model_path)['state_dict']
        print (self.load_state_dict(state_dict, strict=False))

    def forward(self, wavs, phones):
        self.eval()
        with torch.no_grad():
            batch_size = len(wavs)
            speaker_embeddings = []
            for wav in wavs:
                if self.hp.spkr_embedding_path:
                    speaker_embeddings.append(np.load(wav))
                else:
                    wav = normalize(wav) * 0.95
                    wav = torch.FloatTensor(wav).unsqueeze(0)
                    speaker_embedding = self.spkr_embedding({'waveform': wav, 'sample_rate': self.hp.sample_rate})
                    speaker_embeddings.append(speaker_embedding)
            speaker_embeddings = torch.cuda.FloatTensor(np.array(speaker_embeddings))
            norm_spkr = F.normalize(speaker_embeddings, dim=-1)
            speaker_embedding = self.spkr_linear(norm_spkr)
            noise_device = speaker_embedding.device
            low_background_noise = torch.randn(
                batch_size,
                int(self.hp.sample_rate * 5.0),
                device=noise_device
            ) * self.hp.prior_noise_level
            if self.hp.clean_speech_prior:
                base_prior = self.vocoder.encode(low_background_noise)
                prior = base_prior[:, :self.hp.prior_frame]
            else:
                prior = None
            phone_features, phone_masks = [], []
            for phone in phones:
                phone = [self.hp.phoneset.index(ph) for ph in phone if ph != ' ' and ph in self.hp.phoneset]
                phone = np.array(phone)
                phone_features.append(phone)
            #Pad phones
            maxlen = max([len(x) for x in phone_features])
            for i, ph in enumerate(phone_features):
                to_pad = maxlen - len(ph)
                pad = np.zeros([to_pad,], dtype=np.float32)
                pad.fill(self.hp.phoneset.index('<pad>'))
                phone_features[i] = np.concatenate([ph, pad], 0)
                mask = [False] * len(ph)+ [True] * to_pad
                phone_masks.append(mask)
            phone_masks = torch.cuda.BoolTensor(phone_masks)
            phone_features = torch.cuda.LongTensor(phone_features)
            phone_features = self.phone_embedding(phone_features)
            synthetic = self.TTSdecoder.inference_topkp_sampling_batch(
                phone_features, speaker_embedding, phone_masks, prior=prior
            )
            outputs = []
            for i, seq in enumerate(synthetic):
                tokens = seq
                if self.hp.clean_speech_prior and prior is not None:
                    tokens = torch.cat([prior[i, :], tokens], 0)
                audio = self.vocoder(tokens.unsqueeze(0), norm_spkr[i:i+1])
                outputs.append(audio[0, 0].detach().cpu().numpy())
            return outputs
