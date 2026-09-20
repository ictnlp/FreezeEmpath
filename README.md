# FreezeEmpath: Efficient Training for Empathetic Spoken Chatbots with Frozen LLMs

> [Yun Hong](https://hongyun2002.github.io/), [Yan Zhou](https://zhouyan19.github.io/zhouyan/), [Yang Feng*](https://people.ucas.edu.cn/~yangfeng)

Official inference code for our **Findings of ACL 2026** paper.

[![ACL Anthology](https://img.shields.io/badge/ACL_Anthology-2026.findings--acl.846-blue)](https://aclanthology.org/2026.findings-acl.846/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.18159-b31b1b)](https://arxiv.org/abs/2604.18159)
[![Model](https://img.shields.io/badge/Hugging_Face-FreezeEmpath-yellow)](https://huggingface.co/ICTNLP/FreezeEmpath)
[![Decoder](https://img.shields.io/badge/Hugging_Face-Speech_Decoder-yellow)](https://huggingface.co/ICTNLP/indextts2_decoder)

FreezeEmpath is an empathetic spoken chatbot that understands the content and emotional tone of speech and responds with text and expressive speech. It uses a frozen Qwen2.5-7B-Instruct LLM, a shared Whisper-large-v3 speech encoder, separate semantic and emotional features, and a speech decoder followed by an IndexTTS2-based token-to-waveform module.

This release provides pretrained checkpoints, inference code, and Chinese/English examples. Training scripts and training-data preparation are not included.

![FreezeEmpath architecture](asserts/arch.png)

## Installation

Use Linux, Python 3.10, and an NVIDIA GPU with CUDA and BF16 support. The example pipeline processes one utterance at a time. Hardware measurements from the example run are documented in [examples/README.md](examples/README.md).

```bash
git clone https://github.com/ictnlp/FreezeEmpath.git
cd FreezeEmpath
conda create -n freezeempath python=3.10 -y
conda activate freezeempath

# FFmpeg is required to read input audio.
conda install -c conda-forge ffmpeg -y

# Choose a CUDA wheel supported by your NVIDIA driver. This is the tested build.
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Keep `transformers==4.43.4`: the custom speech generation code depends on that version's generation interfaces. No FlashAttention or DeepSpeed installation is required for this pipeline.

## Download checkpoints

```bash
python scripts/download_models.py
```

The downloader fetches [FreezeEmpath](https://huggingface.co/ICTNLP/FreezeEmpath/tree/abeaefa), including **both tokenizers**, the [speech decoder](https://huggingface.co/ICTNLP/indextts2_decoder/tree/2c346af), and the official Whisper-large-v3 checkpoint. It pins the Hugging Face model revisions and reuses existing downloads. Allow roughly 24 GB for checkpoint files, plus environment and cache space. Whisper preparation also loads its checkpoint in CPU memory.

```text
checkpoints/
├── FreezeEmpath/
│   ├── config.json
│   ├── model-00001-of-00004.safetensors  # four shards in total
│   ├── model.safetensors.index.json
│   ├── tokenizer_config.json           # plus vocabulary/tokenizer files
│   └── tts_tokenizer/                  # required for speech generation
├── indextts2_codes2wav/
│   ├── config.yaml
│   ├── semantic_codec.safetensors
│   ├── s2mel.pth
│   ├── bigvgan_generator.pt
│   ├── style.pt
│   ├── prompt_condition.pt
│   └── ref_mel.pt
└── whisper/
    └── large-v3.pt
```

To check local file presence without downloading:

```bash
python scripts/download_models.py --check
```

The decoder uses the BigVGAN weights in this package and the architecture configuration in `configs/`; no further Hugging Face download is needed at inference time. See [third-party notices](THIRD_PARTY_NOTICES.md) for the separate decoder and component licenses.

## Quick start

From the repository root, run the eight included Chinese and English examples:

```bash
CUDA_VISIBLE_DEVICES=0 bash empathetic_dialogue.sh
```

The script runs response generation and then waveform decoding, stopping if either step fails. Results are saved to:

```text
outputs/empathetic_dialogue/
├── answers.jsonl          # input ID, audio path, response text, speech tokens
├── answers.metrics.json  # generation settings, time, and GPU memory
└── wav/
    ├── 0.wav             # 22,050 Hz mono audio; one WAV per input
    ├── ...
    ├── manifest.jsonl    # input ID → response text → WAV filename
    └── metrics.json      # decoder time and GPU memory
```

[Listen to example responses and read their transcripts](examples/README.md).

The default seed is 42 and text temperature is 0.7. Speech tokens and waveform generation also use sampling; results can vary across hardware and library versions.

## Use your own audio

Create a UTF-8 JSONL file with one audio file per line:

```json
{"id": "my-example", "audio": "/absolute/path/to/my_audio.wav"}
```

Only `audio` is required. `id` is optional. The `text` and `emotion` fields in our example manifest are reference metadata and are **not passed to the model**; no transcript or emotion annotation is needed for inference.

Relative audio paths are first resolved from the working directory, then from the manifest's directory. The shell launcher changes to the repository root, so use absolute paths for files outside the repository when in doubt.

```bash
CUDA_VISIBLE_DEVICES=0 bash empathetic_dialogue.sh /path/to/my_examples.jsonl outputs/my_examples
```

FFmpeg decodes and resamples input audio to 16 kHz mono. Inputs are padded or trimmed to 30 seconds; longer recordings produce a warning and only their first 30 seconds are used. Split longer recordings into shorter utterances before running this example.

### Run the two stages separately

```bash
python inference/infer.py \
    --model_path checkpoints/FreezeEmpath \
    --speech_encoder checkpoints/whisper/large-v3.pt \
    --question_file examples/manifest.jsonl \
    --answer_file outputs/answers.jsonl \
    --temperature 0.7 --seed 42 --max_new_tokens 256 --s2s

python inference/codes2wav.py \
    --cfg_path checkpoints/indextts2_codes2wav/config.yaml \
    --model_dir checkpoints/indextts2_codes2wav \
    --input_file outputs/answers.jsonl \
    --output_dir outputs/wav --seed 42
```

`--max_samples 2` limits the first stage to two inputs. Run either entry point with `--help` for its options. The launcher also accepts `CHECKPOINT_DIR`, `WHISPER_MODEL`, `PYTHON`, `SEED`, `TEMPERATURE`, and `MAX_NEW_TOKENS` environment overrides.

## Notes

- This example is an offline, single-turn audio-file interface. It does not expose a live microphone UI, conversation history, or realtime audio playback.
- Output speech uses the fixed voice conditions included in the decoder. The current example does not offer speaker selection or voice cloning.
- CPU-only and Apple MPS inference are not supported by the complete pipeline.
- If CUDA runs out of memory, close other GPU workloads or use a GPU with more free memory. Reducing `MAX_NEW_TOKENS` limits output length; batch size is already one.
- Missing `tts_tokenizer/`, Whisper weights, or decoder files can be identified with `python scripts/download_models.py --check`. Run the downloader to complete the installation.

## Acknowledgements and license

We thank the authors of Qwen, Whisper, LLaMA-Omni2, SLAM-LLM, IndexTTS2, BigVGAN, Amphion/MaskGCT, BLSP-Emo, and VStyle.

FreezeEmpath-specific code is released under [Apache-2.0](LICENSE). Third-party code, decoder weights, and example audio retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). In particular, the IndexTTS2-derived decoder is subject to the included bilibili Model Use License Agreement.

## Citation

```bibtex
@inproceedings{hong-etal-2026-freezeempath,
  title     = {{F}reeze{E}mpath: Efficient Training for Empathetic Spoken Chatbots with Frozen {LLM}s},
  author    = {Hong, Yun and Zhou, Yan and Feng, Yang},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026},
  pages     = {17141--17157},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2026.findings-acl.846},
  url       = {https://aclanthology.org/2026.findings-acl.846/}
}
```
