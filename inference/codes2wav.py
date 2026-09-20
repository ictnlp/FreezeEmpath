import os
import sys

from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import torch
import torchaudio
from transformers import set_seed
import safetensors.torch
import re

import json
import argparse

from indextts.utils.maskgct.models.codec.kmeans.repcodec_model import RepCodec
from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan


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

        semantic_codec = RepCodec(cfg=self.cfg.semantic_codec.conf)
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

        # The published decoder already includes this vocoder's weights.
        # Use the matching bundled architecture instead of an implicit HF download.
        expected_vocoder = "nvidia/bigvgan_v2_22khz_80band_256x"
        if self.cfg.vocoder.name != expected_vocoder:
            raise ValueError(f"Expected vocoder {expected_vocoder}, got {self.cfg.vocoder.name}")
        vocoder_config = ROOT / "configs" / "bigvgan_v2_22khz_80band_256x.json"
        self.bigvgan = bigvgan.BigVGAN(bigvgan.load_hparams_from_json(vocoder_config), use_cuda_kernel=False)
        vocoder_state = torch.load(os.path.join(model_dir, "bigvgan_generator.pt"), map_location="cpu", weights_only=True)
        self.bigvgan.load_state_dict(vocoder_state["generator"])
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()

        # prepare fixed tensor
        self.cache_s2mel_style = torch.load(os.path.join(model_dir, "style.pt"), map_location="cpu", weights_only=True).to(self.device)
        self.cache_s2mel_prompt = torch.load(os.path.join(model_dir, "prompt_condition.pt"), map_location="cpu", weights_only=True).to(self.device)
        self.cache_mel = torch.load(os.path.join(model_dir, "ref_mel.pt"), map_location="cpu", weights_only=True).to(self.device)

    def codes2wav(self, codes, output_path):
        style = self.cache_s2mel_style
        prompt_condition = self.cache_s2mel_prompt
        ref_mel = self.cache_mel

        if not isinstance(codes, str):
            raise ValueError("Speech tokens must be a string such as '<12><34>'.")
        nums = re.findall(r'<(\d+)>', codes)
        if not nums:
            raise ValueError("No speech tokens were generated. Check prediction_units or rerun the response generation.")
        if any(int(x) >= self.cfg.semantic_codec.conf.codebook_size for x in nums):
            raise ValueError("Speech token is outside the semantic codec vocabulary.")
        nums = [int(x) for x in nums]
        codes = torch.tensor(nums, dtype=torch.int64)

        codes = codes.to(self.device)
        chunk_size = 1500
        chunks = [codes[i:i+chunk_size] for i in range(0, len(codes), chunk_size)]

        wavs = []
        sampling_rate = int(self.cfg.s2mel.preprocess_params.sr)
        
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


def main():
    parser = argparse.ArgumentParser(description="Decode FreezeEmpath speech tokens into WAV files.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cfg_path", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with open(args.input_file, encoding="utf-8") as stream:
        data = [json.loads(line) for line in stream if line.strip()]
    if not data:
        raise ValueError("Input file contains no responses.")
    for i, item in enumerate(data):
        units = item.get("prediction_units", item.get("unit"))
        if not isinstance(units, str) or not re.search(r"<(\d+)>", units):
            raise ValueError(f"Response {i} has no speech tokens. Run infer.py with --s2s first.")
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = Token2Wav(cfg_path=args.cfg_path, model_dir=args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the input-to-output mapping next to the generated WAV files.
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for i, item in enumerate(data):
            path = output_dir / f"{i}.wav"
            model.codes2wav(item.get("prediction_units", item.get("unit")), str(path))
            record = {key: value for key, value in item.items() if key not in ("prediction_units", "unit")}
            record["wav"] = path.name
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    metrics = {
        "samples": len(data), "seed": args.seed,
        "elapsed_seconds_including_load": round(time.perf_counter() - started, 2),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2) if torch.cuda.is_available() else None,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(data)} WAV files and manifest.jsonl to {output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
