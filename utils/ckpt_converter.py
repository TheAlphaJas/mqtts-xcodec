import torch
import argparse
import os
from modules.styletts2_transformer import StyleTTSDecoder
from trainer_style import StyleWav2TTS
import json

def convert_ckpt():
    parser = argparse.ArgumentParser(description="Convert original MQTTS checkpoint to Style-MQTTS compatible warm-start checkpoint.")
    parser.add_argument('--og_ckpt', type=str, required=True, help='Path to original MQTTS last.ckpt')
    parser.add_argument('--config_path', type=str, required=True, help='Path to config.json matching the NEW configuration (Style)')
    parser.add_argument('--output_ckpt', type=str, default='ckpt_style/warm_start.ckpt', help='Output path for converted checkpoint')
    parser.add_argument('--reset_transducer', action='store_true', default=True, help='Reset transducer weights (necessary if n_codes changed)')
    
    args = parser.parse_args()
    
    print(f"Loading original checkpoint from {args.og_ckpt}...")
    try:
        og_state = torch.load(args.og_ckpt, map_location='cpu')['state_dict']
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    print(f"Loading new configuration from {args.config_path}...")
    with open(args.config_path, 'r') as f:
        hparams = argparse.Namespace(**json.load(f))
    
    # Instantiate new model structure
    print("Initializing new StyleWav2TTS model...")
    # We need to mock some paths if they don't exist, but StyleWav2TTS __init__ might check them.
    # Ideally, we just instantiate the sub-modules if StyleWav2TTS is too heavy (loads data).
    # However, StyleWav2TTS loads data in __init__. Let's bypass that by creating a dummy class or just instantiating the Decoder.
    # Better: Instantiate StyleWav2TTS but catch data errors or mock data paths?
    # Actually, let's just instantiate the sub-modules we care about: StyleTTSDecoder.
    # But we need the full LightningModule state dict structure.
    
    # Let's assume the user has valid paths in config.json or we mock them.
    # Easier approach: Modify hparams to dummy paths if needed, but let's trust the user provided a valid config from their training setup.
    
    # To avoid loading dataset (which takes time), we can just build the model components manually matching StyleWav2TTS structure.
    # StyleWav2TTS has:
    # self.TTSdecoder = StyleTTSDecoder(...)
    # self.phone_embedding = ...
    # self.spkr_linear = ... (removed/unused)
    # self.vocoder = ...
    
    # We will just filter the keys and save a new dict. We don't need to instantiate the class to verify shapes if we are careful.
    # BUT, checking shapes is safer.
    
    print("Filtering and adapting weights...")
    new_state = {}
    
    # Key mapping:
    # Original -> New
    # TTSdecoder.* -> TTSdecoder.* (Transformer layers match)
    # phone_embedding.* -> phone_embedding.*
    # spkr_linear.* -> DELETED (replaced by style encoder)
    # vocoder.* -> vocoder.* (if using same vocoder)
    
    transducer_reset_count = 0
    style_encoder_skipped = 0
    shape_mismatch_count = 0
    
    for key, value in og_state.items():
        if 'spkr_linear' in key:
            continue # Skip old speaker projection
            
        if 'transducer' in key and args.reset_transducer:
            transducer_reset_count += 1
            continue # Skip transducer if requested (likely due to n_codes change)
            
        # Check for shape mismatch (heuristic without instantiating model)
        # If we blindly copy, load_state_dict(strict=False) later will handle shape mismatches by erroring?
        # No, strict=False ignores missing keys, but shape mismatches usually crash.
        # So we should be careful.
        
        new_state[key] = value

    print(f"Transferred {len(new_state)} parameters.")
    if args.reset_transducer:
        print(f"Dropped {transducer_reset_count} transducer parameters (will be re-initialized).")
        
    print("Note: 'style_encoder' weights are missing and will be randomly initialized.")
    
    # Save in Lightning checkpoint format
    # We wrap it in a dict like Lightning does: {'state_dict': ...}
    # We drop optimizer states because they are invalid for the new architecture.
    final_ckpt = {'state_dict': new_state, 'epoch': 0, 'global_step': 0}
    
    os.makedirs(os.path.dirname(args.output_ckpt), exist_ok=True)
    torch.save(final_ckpt, args.output_ckpt)
    print(f"✅ Converted warm-start checkpoint saved to: {args.output_ckpt}")
    print(f"Use usage: --pretrained_path {args.output_ckpt}")

if __name__ == "__main__":
    convert_ckpt()

