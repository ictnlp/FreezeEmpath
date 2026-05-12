import os
import sys

sys.path.append(".")


import torch
import torchaudio
from transformers import SeamlessM4TFeatureExtractor
from huggingface_hub import hf_hub_download
import safetensors
import re

import json
import argparse

from indextts.utils.maskgct_utils import build_semantic_codec
from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram


from omegaconf import OmegaConf

class Token2Wav:
    def __init__(
            self, cfg_path, model_dir, device=None
    ):
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir

        semantic_codec = build_semantic_codec(self.cfg.semantic_codec.conf)
        semantic_code_ckpt = os.path.join(self.model_dir, self.cfg.semantic_codec.checkpoint)
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device)
        self.semantic_codec.eval()

        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device)
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.s2mel.eval()

        bigvgan_name = self.cfg.vocoder.name
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()

        mel_fn_args = {
            "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

        # prepare fixed tensor
        self.cache_s2mel_style = torch.load(os.path.join(model_dir, "style.pt")).to(self.device)
        self.cache_s2mel_prompt = torch.load(os.path.join(model_dir, "prompt_condition.pt")).to(self.device)
        self.cache_mel = torch.load(os.path.join(model_dir, "ref_mel.pt")).to(self.device)

    def codes2wav(self, codes, output_path):
        style = self.cache_s2mel_style
        prompt_condition = self.cache_s2mel_prompt
        ref_mel = self.cache_mel

        nums = re.findall(r'<(\d+)>', codes)
        nums = [int(x) for x in nums]
        codes = torch.tensor(nums, dtype=torch.int64)

        codes = codes.to(self.device)
        chunk_size = 1500
        chunks = [codes[i:i+chunk_size] for i in range(0, len(codes), chunk_size)]

        wavs = []
        sampling_rate = 22050
        
        for chunk in chunks:
            with torch.no_grad():
                chunk = chunk.unsqueeze(0)
                code_lens = torch.LongTensor([chunk.size(-1)])
                code_lens = code_lens.to(self.device)

                dtype = None
                with torch.amp.autocast(chunk.device.type, enabled=dtype is not None, dtype=dtype):
                    diffusion_steps = 25
                    inference_cfg_rate = 0.7

                    S_infer = self.semantic_codec.quantizer.vq2emb(chunk.unsqueeze(1))
                    S_infer = S_infer.transpose(1, 2)

                    target_lengths = (code_lens * 1.72).long()
                    
                    cond = self.s2mel.models['length_regulator'](S_infer,
                                                                    ylens=target_lengths,
                                                                    n_quantizers=3,
                                                                    f0=None)[0]
                    cat_condition = torch.cat([prompt_condition, cond], dim=1)
                    vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                                    torch.LongTensor([cat_condition.size(1)]).to(
                                                                        cond.device),
                                                                    ref_mel, style, None, diffusion_steps,
                                                                    inference_cfg_rate=inference_cfg_rate)
                    vc_target = vc_target[:, :, ref_mel.size(-1):]

                    wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                wavs.append(wav.cpu())

        wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=200)
        wav = torch.cat(wavs, dim=1)

        wav = wav.cpu()
        if output_path:
            if os.path.isfile(output_path):
                os.remove(output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            
            return output_path
        else:
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Insert silences between generated segments.
        wavs: List[torch.tensor]
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)

        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)

        return wavs_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_dir")
    parser.add_argument("--cfg_path")
    parser.add_argument("--model_dir")
    args = parser.parse_args()
    model = Token2Wav(cfg_path=args.cfg_path, model_dir=args.model_dir)
    input_file = args.input_file
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data = []
    with open(input_file, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    for i, item in enumerate(data):
        codes = item["prediction_units"] if "prediction_units" in item else item['unit']
        spk_audio_prompt = 'index-tts/spk_audio_prompt/voice_11.wav'
        output_path = f"{output_dir}/{i}.wav"
        model.codes2wav(codes=codes, output_path=output_path)