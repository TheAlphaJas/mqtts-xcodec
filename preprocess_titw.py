import os
import numpy as np
import soundfile as sf
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from dp.phonemizer import Phonemizer
import pyloudnorm as pyln
import torchaudio
import warnings
warnings.filterwarnings("ignore")

def process_json_file(json_path, audio_base_dir, output_audio_dir, phonemizer, meter):
    """
    Processes a JSON file containing TITW-easy data entries.
    Each entry is processed to normalize audio, generate phonemes, and prepare metadata.
    """
    output_data = {}
    
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        # Assuming the JSON file contains a list of dictionaries
        data = json.load(f)

    for entry in tqdm(data, desc=f"Processing {Path(json_path).name}"):
        try:
            # Construct audio file path using the base directory
            full_audio_path = Path(audio_base_dir) / Path(entry['audio_filepath'])
            
            if not full_audio_path.exists():
                print(f"Warning: Audio file not found, skipping: {full_audio_path}")
                continue

            output_filename = full_audio_path.name
            output_audio_path = output_audio_dir / output_filename

            # Load and resample audio to 16kHz if necessary, as expected by MQTTS
            audio, sr = torchaudio.load(str(full_audio_path))
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                audio = resampler(audio)
            sr = 16000

            # Convert to mono numpy array for processing
            mono_audio_np = audio.mean(0).numpy()

            # Skip silent or invalid audio to prevent errors
            if np.std(mono_audio_np) < 1e-4:
                print(f"Skipping silent/empty audio file: {output_filename}")
                continue
            
            # Apply loudness normalization and fade-in/fade-out, same as original script
            loudness = meter.integrated_loudness(mono_audio_np)
            normalized_audio_np = pyln.normalize.loudness(mono_audio_np, loudness, -20.0)
            
            fade_len = 1600 # 100ms fade at 16kHz
            if normalized_audio_np.shape[0] > 2 * fade_len:
                fade_in = np.linspace(0.0, 1.0, fade_len)
                fade_out = np.linspace(1.0, 0., fade_len)
                normalized_audio_np[:fade_len] *= fade_in
                normalized_audio_np[-fade_len:] *= fade_out

            # Save the processed audio file
            sf.write(str(output_audio_path), normalized_audio_np, sr)

            # Process text and generate phonemes
            text = entry['text'].lower()
            phonemes = phonemizer(text, lang='en_us').replace('[', ' ').replace(']', ' ')
            
            # Calculate duration from the final processed audio
            duration = normalized_audio_np.shape[0] / sr

            # Store metadata for the final JSON needed by the transformer training
            output_data[output_filename] = {
                'text': text,
                'duration': duration,
                'phoneme': phonemes,
                'spk_id': entry['spk_id']
            }
        except Exception as e:
            print(f"Error processing entry with audio file '{entry.get('audio_filepath', 'N/A')}': {e}")
    
    return output_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess TITW-easy dataset for MQTTS training.")
    parser.add_argument('--train_json', type=str, required=True, help="Path to the training JSON file (e.g., training.json).")
    parser.add_argument('--val_json', type=str, required=True, help="Path to the validation JSON file (e.g., validation.json).")
    parser.add_argument('--outputdir', type=str, required=True, help="Directory to save the processed files (e.g., 'datasets').")
    parser.add_argument('--audio_base_dir', type=str, default='.', help="Base directory where the audio files are located.")
    args = parser.parse_args()

    # --- 1. Setup ---
    # Ensure the phonemizer model is downloaded as per the MQTTS README
    phonemizer_checkpoint = 'en_us_cmudict_forward.pt'
    if not os.path.exists(phonemizer_checkpoint):
        print(f"Phonemizer checkpoint not found at '{phonemizer_checkpoint}'.")
        print("Please download it by running: wget https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_forward.pt")
        exit(1)
        
    phonemizer = Phonemizer.from_checkpoint(phonemizer_checkpoint)
    
    outputdir = Path(args.outputdir)
    outputaudiodir = outputdir / 'audios'
    outputaudiodir.mkdir(exist_ok=True, parents=True)
    
    # Initialize loudness meter for 16kHz audio
    meter = pyln.Meter(16000)

    # --- 2. Process Data ---
    train_output_data = process_json_file(args.train_json, args.audio_base_dir, outputaudiodir, phonemizer, meter)
    val_output_data = process_json_file(args.val_json, args.audio_base_dir, outputaudiodir, phonemizer, meter)
    
    # --- 3. Save JSON files for Transformer Training ---
    train_json_path = outputdir / 'train.json'
    dev_json_path = outputdir / 'dev.json'
    
    print(f"Saving processed training metadata to {train_json_path}...")
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_output_data, f, indent=2)
        
    print(f"Saving processed validation metadata to {dev_json_path}...")
    with open(dev_json_path, 'w', encoding='utf-8') as f:
        json.dump(val_output_data, f, indent=2)

    # --- 4. Save .txt files for Quantizer Training ---
    train_txt_path = outputdir / 'training.txt'
    val_txt_path = outputdir / 'validation.txt'

    print(f"Saving training file list to {train_txt_path}...")
    with open(train_txt_path, 'w', encoding='utf-8') as f:
        for filename in train_output_data.keys():
            f.write(f"{Path(filename).stem}\n")

    print(f"Saving validation file list to {val_txt_path}...")
    with open(val_txt_path, 'w', encoding='utf-8') as f:
        for filename in val_output_data.keys():
            f.write(f"{Path(filename).stem}\n")

    print("Preprocessing complete.")

