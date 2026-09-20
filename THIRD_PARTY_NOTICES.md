# Third-party code, models, and examples

The Apache-2.0 license in the repository root applies to the FreezeEmpath-specific code. Third-party code, model weights, and example audio retain their respective licenses and copyright notices; the root license does not replace them.

| Component | Source | License / notice |
| --- | --- | --- |
| Speech-language modeling and generation | [LLaMA-Omni2](https://github.com/ictnlp/LLaMA-Omni2), [LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni), [LLaVA](https://github.com/haotian-liu/LLaVA) | Apache-2.0; preserve the existing Haotian Liu and Qingkai Fang notices |
| Whisper encoder wrapper | [SLAM-LLM](https://github.com/ddlBoJack/SLAM-LLM) | Apache-2.0; source attribution is retained in the encoder file |
| Generation utilities | [Transformers 4.43.4](https://github.com/huggingface/transformers/tree/v4.43.4) | [Apache-2.0](licenses/Apache-2.0.txt) |
| Token-to-waveform pipeline and bundled IndexTTS code | [IndexTTS2](https://github.com/index-tts/index-tts) | [bilibili Model Use License Agreement](licenses/IndexTTS2.txt), except components carrying their own upstream licenses |
| Vocoder, activation layers, and inference configuration | [NVIDIA BigVGAN](https://github.com/NVIDIA/BigVGAN), [HiFi-GAN](https://github.com/jik876/hifi-gan) | [BigVGAN MIT](licenses/BigVGAN.txt), [HiFi-GAN MIT](licenses/HiFi-GAN.txt) |
| Semantic codec / MaskGCT components | [Amphion](https://github.com/open-mmlab/Amphion) | [MIT](licenses/Amphion.txt) |
| Transformer inference components | [GPT-fast](https://github.com/meta-pytorch/gpt-fast) | [BSD-3-Clause](licenses/GPT-fast.txt) |
| Convolutional audio components | [Encodec](https://github.com/facebookresearch/encodec) | [MIT](licenses/Encodec.txt) |
| DAC components | [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) | [MIT](licenses/DAC.txt) |
| Base language model and speech decoder initialization | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | Refer to the respective model repositories; Apache-2.0 |
| Speech encoder weights | [OpenAI Whisper](https://github.com/openai/whisper) | MIT; downloaded separately from the official source |

`configs/bigvgan_v2_22khz_80band_256x.json` contains the inference architecture fields from NVIDIA's configuration for [this checkpoint](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x/tree/633ff708ed5b74903e86ff1298cf4a98e921c513). The vocoder weights are loaded from the existing decoder package; this repository does not distribute additional model weights.

## IndexTTS2 decoder

The decoder package is derived from IndexTTS2 and includes fixed speaker-conditioning tensors. Its source license is included in [licenses/IndexTTS2.txt](licenses/IndexTTS2.txt). An Apache-2.0 metadata label on a model-hosting page does not replace the upstream terms.

Any modifications made to the original model in this Derivative Work are not endorsed, warranted, or guaranteed by the original right-holder of the original model, and the original right-holder disclaims all liability related to this Derivative Work.

The original development copy contains the same named bilibili Model Use License Agreement but no Git metadata identifying an exact IndexTTS commit. The upstream license copy was retrieved on September 20, 2026. Original file-level notices remain in place.

## Example audio

- `examples/audio/VStyle-empathy/` contains the four existing Chinese examples from the implicit-empathy portion of [VStyle](https://github.com/alibaba/vstyle), distributed under its [MIT license](licenses/VStyle.txt).
- `examples/audio/SpeechAlpaca/` contains the four existing English SpeechAlpaca examples released with [BLSP-Emo](https://github.com/cwang621/blsp-emo). See that project's SpeechAlpaca release for the dataset source and applicable terms; the FreezeEmpath code license does not relicense this audio.
- `examples/generated/` contains FreezeEmpath responses to these existing inputs, using the released decoder's fixed voice conditions. No claim of ownership of the source speaker identity is made.
