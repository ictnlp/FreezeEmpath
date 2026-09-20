---
language:
- en
- zh
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: transformers
tags:
- speech
- empathetic-dialogue
- audio-to-audio
- custom-code
---

# FreezeEmpath

Pretrained model for **FreezeEmpath: Efficient Training for Empathetic Spoken Chatbots with Frozen LLMs**, Findings of ACL 2026, by Yun Hong, Yan Zhou, and Yang Feng.

[Paper](https://arxiv.org/abs/2604.18159) · [Code and installation](https://github.com/ictnlp/FreezeEmpath) · [Example responses](https://github.com/ictnlp/FreezeEmpath/blob/main/examples/README.md)

FreezeEmpath accepts Chinese or English speech and generates empathetic text and speech-token responses. It combines Qwen2.5-7B-Instruct with a Whisper-large-v3 speech encoder, a semantic adapter, an emotion extractor, and a speech decoder. The base LLM was kept frozen during training.

## Usage

This checkpoint requires the custom model classes in the GitHub repository; a bare Transformers `pipeline()` call is not supported. Install the pinned dependencies and use:

```bash
python scripts/download_models.py
CUDA_VISIBLE_DEVICES=0 bash empathetic_dialogue.sh
```

Download this entire repository, including all four weight shards, the main tokenizer, and `tts_tokenizer/`. The downloader also obtains Whisper-large-v3 and the separate [IndexTTS2-derived waveform decoder](https://huggingface.co/ICTNLP/indextts2_decoder).

The supported example uses Linux, Python 3.10, CUDA/BF16, and Transformers 4.43.4. It handles single-turn audio, one sample at a time, and pads/trims input to 30 seconds. No transcript or emotion annotation is required. See the GitHub examples for measured hardware usage.

## License and attribution

This model repository retains its Apache-2.0 license declaration. The separately downloaded waveform decoder has its own upstream license and is not covered by this declaration. See the GitHub [third-party notices](https://github.com/ictnlp/FreezeEmpath/blob/main/THIRD_PARTY_NOTICES.md) and the source model cards for Qwen and Whisper.
