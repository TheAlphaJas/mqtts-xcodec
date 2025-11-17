import os
import json
import random

# --- Configuration ---
# Adjust these paths to match your system
titw_metadata_path = './titw_easy_metadata/titw_easy_metadata/bonafide_metadata_cfg_v3'
titw_audio_path = './titw-easy-audio/titw-easy-audio/'
output_dir = './Data-out'

# Splitting configuration
SOURCE_DATA_SPLIT = 'dev' # The split to be partitioned into train/validation
VALIDATION_SET_RATIO = 0.15 # Use 15% of the data for validation
RANDOM_SEED = 42 # For reproducible splits

os.makedirs(output_dir, exist_ok=True)
# Removed the creation of the 'audios' directory as symlinks are no longer used.


# --- Helper Functions ---
def read_kaldi_file(filepath):
    """Reads a Kaldi-style file and returns a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                data[parts[0]] = parts[1]
    return data

def create_split_files(utt_ids, split_name, source_audio_dir, transcripts, utt2spk):
    """Processes a list of utterance IDs and writes the corresponding json and txt files."""
    print(f"\nProcessing '{split_name}' split with {len(utt_ids)} utterances...")
    json_data = []
    txt_lines = []

    for utt_id in utt_ids:
        # Construct the original audio file path
        source_audio_path = os.path.join(source_audio_dir, f'{utt_id}.wav')
        
        # Verify the audio file exists before adding it to the manifest
        if not os.path.exists(source_audio_path):
            print(f"Warning: Audio file not found and skipped: {source_audio_path}")
            continue

        # Prepare JSON entry using the original file path
        json_entry = {
            'audio_filepath': source_audio_path,
            'text': transcripts[utt_id],
            'spk_id': utt2spk[utt_id]
        }
        json_data.append(json_entry)

        # Prepare TXT line (format: utterance_id, as requested previously)
        txt_lines.append(utt_id)

    # Write JSON file (for transformer training)
    json_output_path = os.path.join(output_dir, f'{split_name}.json')
    with open(json_output_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"Successfully created {json_output_path} with {len(json_data)} entries.")

    # Write TXT file (for quantizer training)
    txt_output_path = os.path.join(output_dir, f'{split_name}.txt')
    with open(txt_output_path, 'w') as f:
        f.write('\n'.join(txt_lines))
    print(f"Successfully created {txt_output_path} with {len(txt_lines)} lines.")


# --- Load Metadata ---
print("Loading TITW metadata files...")
wav_scp = read_kaldi_file(os.path.join(titw_metadata_path, 'wav.scp'))
transcripts = read_kaldi_file(os.path.join(titw_metadata_path, 'text'))
utt2spk = read_kaldi_file(os.path.join(titw_metadata_path, 'utt2spk'))
print(f"Loaded {len(wav_scp)} audio paths, {len(transcripts)} transcripts, and {len(utt2spk)} utterance-to-speaker mappings.")


# --- Process and Split the Source Data ---
print(f"\nPreparing to split the '{SOURCE_DATA_SPLIT}' set into training and validation...")
source_audio_dir = os.path.join(titw_audio_path, SOURCE_DATA_SPLIT)

if not os.path.exists(source_audio_dir):
    print(f"Error: Source directory not found: {source_audio_dir}. Halting execution.")
else:
    # Find all utterance IDs in the source directory that have complete metadata
    all_utt_ids = {os.path.splitext(filename)[0] for filename in os.listdir(source_audio_dir)}
    valid_utt_ids = [
        utt_id for utt_id in all_utt_ids
        if utt_id in wav_scp and utt_id in transcripts and utt_id in utt2spk
    ]
    print(f"Found {len(valid_utt_ids)} valid utterances in the '{SOURCE_DATA_SPLIT}' directory.")

    # Shuffle and split the utterance IDs into training and validation sets
    random.seed(RANDOM_SEED)
    random.shuffle(valid_utt_ids)

    split_index = int(len(valid_utt_ids) * (1 - VALIDATION_SET_RATIO))
    train_utt_ids = sorted(valid_utt_ids[:split_index])
    val_utt_ids = sorted(valid_utt_ids[split_index:])
    
    print(f"Splitting data into {len(train_utt_ids)} training samples and {len(val_utt_ids)} validation samples.")

    # Create files for the training split ('training.json' and 'training.txt')
    create_split_files(train_utt_ids, 'training', source_audio_dir, transcripts, utt2spk)
    
    # Create files for the validation split ('validation.json' and 'validation.txt')
    create_split_files(val_utt_ids, 'validation', source_audio_dir, transcripts, utt2spk)

    print("\nData preparation complete.")
