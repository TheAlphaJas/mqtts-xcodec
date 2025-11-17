from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import json
import os

import numpy as np
import torch
from librosa.util import normalize
from tqdm import tqdm
from transformers import AutoFeatureExtractor, XcodecModel

from meldataset import MAX_WAV_VALUE, load_wav


class XCodecQuantizer:
    def __init__(self, model_id: str, bandwidth: float, device: torch.device):
        self.model = XcodecModel.from_pretrained(model_id)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.bandwidth = bandwidth
        self.device = device
        self.sample_rate = getattr(self.feature_extractor, "sampling_rate", 16000)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, audio: np.ndarray) -> np.ndarray:
        inputs = self.feature_extractor(
            raw_audio=audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        input_values = inputs["input_values"].to(self.device)
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)
        codes = self.model.encode(input_values, bandwidth=self.bandwidth).audio_codes
        return codes.squeeze(0).cpu().numpy()


def quantize_dataset(a, device: torch.device):
    codec = XCodecQuantizer(a.model_id, a.bandwidth, device)

    with open(a.input_json, 'r') as f:
        metadata = json.load(f)

    with torch.no_grad():
        for filename in tqdm(list(metadata.keys())):
            wav_path = os.path.join(a.input_wav_dir, filename)
            audio, sampling_rate = load_wav(wav_path)
            if sampling_rate != codec.sample_rate:
                raise ValueError(
                    f"Sampling rate mismatch for {wav_path}: expected {codec.sample_rate}, got {sampling_rate}"
                )
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio.astype(np.float32)) * 0.95
            codes = codec.encode(audio)
            metadata[filename]['quantization'] = codes.tolist()

    with open(a.output_json, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    print('Initializing X-Codec quantization pipeline...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', default='../datasets/train.json')
    parser.add_argument('--input_wav_dir', default='../datasets/audios')
    parser.add_argument('--output_json', default='../datasets/train_q.json')
    parser.add_argument('--model_id', default='hf-audio/xcodec-wavlm-more-data')
    parser.add_argument('--bandwidth', type=float, default=2.0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.manual_seed(0)
    if device.type == 'cuda':
        torch.cuda.manual_seed(0)

    quantize_dataset(args, device)


if __name__ == '__main__':
    main()

