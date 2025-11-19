# MQTTS
 - Official implementation for the paper: A Vector Quantized Approach for Text to Speech Synthesis on Real-World Spontaneous Speech.
 - Audio samples (40 each system) can be accessed at [here](https://cmu.box.com/s/ktbk9pi04e2z1dlyepkkw69xcu9w91dj).
 - Quick demo can be accessed [here](https://b04901014.github.io/MQTTS/) (Some are still TODO).
 - Paper appendix is [here](https://cmu.box.com/s/7ghw0bgkbqv5e7hu5jsznhlzuo4rexgx).
## Setup the environment
1. Setup conda environment:
```
conda create --name mqtts python=3.9
conda activate mqtts
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
Note: For CUDA 11.8, use `pytorch-cuda=11.8`. For CUDA 12.1, use `pytorch-cuda=12.1`. For CPU-only, use `cpuonly` instead of `pytorch-cuda`.
(Update) You may need to create an access token to use the speaker embedding of pyannote as they updated their policy.
If that's the case follow the [pyannote repo](https://github.com/pyannote/pyannote-audio) and change every `Inference("pyannote/embedding", window="whole")` accordingly.

2. Download the pretrained phonemizer checkpoint
```
wget https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_forward.pt
```

## Preprocess the dataset
1. Get the GigaSpeech dataset from the [official repo](https://github.com/SpeechColab/GigaSpeech)
2. Install [FFmpeg](https://ffmpeg.org), then
```
conda install ffmpeg=4.3=hf484d3e_0
conda update ffmpeg
```
3. Run python script
```
python preprocess.py --giga_speech_dir GIGASPEECH --outputdir datasets 
```

## Generate codec tokens with X-Codec
We now rely on the pretrained [X-Codec](https://huggingface.co/docs/transformers/en/model_doc/xcodec) model instead of
training our own quantizer. Run the helper script to attach codec tokens to each metadata entry:
```
python quantizer/get_labels.py --input_json ../datasets/train.json \
                               --input_wav_dir ../datasets/audios \
                               --output_json ../datasets/train_q.json \
                               --model_id hf-audio/xcodec-wavlm-more-data \
                               --bandwidth 2.0

python quantizer/get_labels.py --input_json ../datasets/dev.json \
                               --input_wav_dir ../datasets/audios \
                               --output_json ../datasets/dev_q.json \
                               --model_id hf-audio/xcodec-wavlm-more-data \
                               --bandwidth 2.0
```
The script automatically downloads the requested checkpoint from the Hugging Face Hub.

## Train the transformer (below an example for the 100M version)
```
cd ..
mkdir ckpt
python train.py \
     --distributed \
     --saving_path ckpt/ \
     --sampledir logs/ \
     --codec_model_id hf-audio/xcodec-wavlm-more-data \
     --codec_bandwidth 2.0 \
     --datadir datasets/audios \
     --metapath datasets/train_q.json \
     --val_metapath datasets/dev_q.json \
     --n_codes 512 \
     --source_n_codes 1024 \
     --use_repetition_token \
     --ar_layer 4 \
     --ar_ffd_size 1024 \
     --ar_hidden_size 256 \
     --ar_nheads 4 \
     --speaker_embed_dropout 0.05 \
     --enc_nlayers 6 \
     --dec_nlayers 6 \
     --ffd_size 3072 \
     --hidden_size 768 \
     --nheads 12 \
     --batch_size 200 \
     --precision bf16-mixed \
     --training_step 800000 \
     --layer_norm_eps 1e-05
```
You can view the progress using:
```
tensorboard --logdir logs/
```

## Run batched inference

You'll have to change `speaker_to_text.json`, it's just an example.

```
mkdir infer_samples
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --phonemizer_dict_path en_us_cmudict_forward.pt \
    --model_path ckpt/last.ckpt \
    --config_path ckpt/config.json \
    --input_path speaker_to_text.json \
    --outputdir infer_samples \
    --batch_size {batch_size} \
    --top_p 0.8 \
    --min_top_k 2 \
    --max_output_length {Maximum Output Frames to prevent infinite loop} \
    --phone_context_window 3 \
    --clean_speech_prior
```

### Pretrained checkpoints

1. Quantizer (X-Codec): handled automatically via the Hugging Face Hub (`hf-audio/xcodec-wavlm-more-data`).

2. Transformer (100M version) (put it under `ckpt/`): [model](https://cmu.box.com/s/xuen9o8wxsmyaz32a65fu25cz92a2jei), [config](https://cmu.box.com/s/hvv06w3yr8mob4csjjaigu5szq2qcjab)
