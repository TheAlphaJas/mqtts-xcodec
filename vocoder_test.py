import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from modules.vocoder import Vocoder


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(path)
    if sr != sample_rate:
        raise ValueError(f"Expected {sample_rate} Hz audio but found {sr} Hz in {path}")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    audio = np.clip(audio * 0.95, -0.995, 0.995)
    return audio


def parse_args():
    parser = argparse.ArgumentParser(description="XCodec vocoder encode/decode sanity check.")
    parser.add_argument('--datadir', type=str, required=True, help='Directory containing dev audio files.')
    parser.add_argument('--metapath', type=str, required=True, help='Metadata JSON describing dev split.')
    parser.add_argument('--outputdir', type=str, default='vocoder_test', help='Directory to write comparison wavs.')
    parser.add_argument('--codec_model_id', type=str, default='hf-audio/xcodec-wavlm-more-data', help='HF model id.')
    parser.add_argument('--codec_bandwidth', type=float, default=2.0, help='Codec bandwidth for encoding.')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Expected sampling rate.')
    parser.add_argument('--k', type=int, default=10, help='Number of dev samples to evaluate.')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed for sample selection.')
    parser.add_argument('--device', type=str, default=None, help='Torch device override (e.g. cuda:0).')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    with open(args.metapath, 'r') as f:
        metadata = json.load(f)

    entries = list(metadata.keys())
    if not entries:
        raise ValueError(f"No entries found in metadata {args.metapath}")

    num_samples = min(args.k, len(entries))
    selected = random.sample(entries, num_samples)

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    vocoder = Vocoder(args.codec_model_id, args.codec_bandwidth, sample_rate=args.sample_rate)
    vocoder = vocoder.to(device)
    vocoder.eval()

    print(f"[VocoderTest] Using device: {device}, codec model: {args.codec_model_id}, bandwidth: {args.codec_bandwidth}")
    print(f"[VocoderTest] Sampling {num_samples} / {len(entries)} examples from {args.metapath}")

    for idx, rel_path in enumerate(selected, 1):
        audio_path = Path(args.datadir) / rel_path
        if not audio_path.exists():
            print(f"[Warning] Skipping missing file: {audio_path}")
            continue
        audio = load_audio(audio_path, args.sample_rate)
        codes = vocoder.encode(audio)
        reconstructed = vocoder(codes).squeeze(0).detach().cpu().numpy()
        reconstructed = np.clip(reconstructed, -0.995, 0.995)

        stem = Path(rel_path).stem
        orig_path = output_dir / f"{idx:02d}_{stem}_orig.wav"
        recon_path = output_dir / f"{idx:02d}_{stem}_recon.wav"

        sf.write(orig_path, audio, args.sample_rate)
        sf.write(recon_path, reconstructed, args.sample_rate)

        print(f"[VocoderTest] Saved {orig_path.name} and {recon_path.name}")

    print(f"[VocoderTest] Completed. Files saved to {output_dir.resolve()}")


if __name__ == '__main__':
    main()

