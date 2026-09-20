"""Generate empathetic responses from a JSONL audio manifest (one utterance at a time)."""
import argparse
import json
from pathlib import Path
import sys
import time
import warnings

import torch
import whisper
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import Omni2Speech2SQwen2ForCausalLM, Omni2SpeechQwen2ForCausalLM
from constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, EMOTION_TOKEN_INDEX, DEFAULT_EMOTION_TOKEN

SYSTEM_PROMPT = (
    "You are an empathetic spoken chatbot. Please provide a very short but helpful "
    "response to the user with empathy toward the user's emotional tone."
)


def read_questions(path):
    path = Path(path).expanduser().resolve()
    questions = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(item, dict) or not isinstance(item.get("audio"), str):
                raise ValueError(f"{path}:{number}: expected an object with an 'audio' path")
            audio = Path(item["audio"]).expanduser()
            # Keep compatibility with the repository's existing manifests.
            if not audio.is_absolute() and not audio.is_file():
                audio = path.parent / audio
            if not audio.is_file():
                raise FileNotFoundError(f"{path}:{number}: audio not found: {item['audio']}")
            questions.append((item, audio.resolve()))
    if not questions:
        raise ValueError(f"No audio examples found in {path}")
    return questions


def load_pretrained_model(model_path, speech_encoder, s2s):
    model_path = Path(model_path).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing {model_path / 'config.json'}. Run scripts/download_models.py first.")
    config = AutoConfig.from_pretrained(str(model_path))
    # Older released configs contain a path relative to the development project.
    config.speech_encoder = speech_encoder
    config.tts_tokenizer = str(model_path / "tts_tokenizer")
    if s2s and not (model_path / "tts_tokenizer" / "tokenizer_config.json").is_file():
        raise FileNotFoundError("The model download must include the tts_tokenizer/ directory.")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
    model_cls = Omni2Speech2SQwen2ForCausalLM if s2s else Omni2SpeechQwen2ForCausalLM
    model = model_cls.from_pretrained(str(model_path), config=config, torch_dtype=torch.bfloat16)
    return tokenizer, model.cuda().eval()


def eval_model(args):
    questions = read_questions(args.question_file)
    if args.max_samples:
        questions = questions[:args.max_samples]
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Inference requires an NVIDIA GPU with CUDA and BF16 support.")
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer, model = load_pretrained_model(args.model_path, args.speech_encoder, args.s2s)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN}(with a {DEFAULT_EMOTION_TOKEN} emotional tone)."},
    ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    for token, index in [(DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX), (DEFAULT_EMOTION_TOKEN, EMOTION_TOKEN_INDEX)]:
        if token not in tokenizer.get_vocab():
            raise ValueError(f"Checkpoint tokenizer is missing {token}")
        input_ids[input_ids == tokenizer.convert_tokens_to_ids(token)] = index
    input_ids = input_ids.cuda()
    output_path = Path(args.answer_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for item, audio_path in tqdm(questions, desc="Generating responses"):
            audio = whisper.load_audio(str(audio_path))
            if len(audio) > whisper.audio.N_SAMPLES:
                warnings.warn(f"{item['audio']}: only the first 30 seconds will be used.")
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=128)
            speech = mel.T.unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
            lengths = torch.tensor([speech.shape[1]], device="cuda")
            with torch.inference_mode():
                result = model.generate(
                    input_ids, attention_mask=torch.ones_like(input_ids),
                    speech=speech, speech_lengths=lengths,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p, top_k=args.top_k, num_beams=1,
                    max_new_tokens=args.max_new_tokens, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            output_ids, units = result if args.s2s else (result, None)
            prediction = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            record = {"question_id": item.get("id", item["audio"]), "audio": item["audio"], "prediction": prediction}
            if args.s2s:
                record["prediction_units"] = units
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
    torch.cuda.synchronize()
    metrics = {
        "samples": len(questions), "seed": args.seed, "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens, "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "elapsed_seconds_including_load": round(time.perf_counter() - started, 2),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }
    output_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(questions)} responses to {output_path}")
    print(json.dumps(metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="checkpoints/FreezeEmpath")
    parser.add_argument("--speech_encoder", default="large-v3", help="Whisper model name or path to large-v3.pt")
    parser.add_argument("--question_file", default="examples/manifest.jsonl")
    parser.add_argument("--answer_file", default="outputs/answers.jsonl")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_beams", type=int, choices=[1], default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    # Compatibility with the original CLI; only this frontend matches the checkpoint.
    parser.add_argument("--input_type", choices=["mel"], default="mel")
    parser.add_argument("--mel_size", type=int, choices=[128], default=128)
    parser.add_argument("--s2s", action="store_true", help="Generate speech tokens as well as text")
    args = parser.parse_args()
    if args.temperature < 0 or args.max_new_tokens < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("temperature must be nonnegative; max_new_tokens and max_samples must be positive")
    return args


if __name__ == "__main__":
    eval_model(parse_args())
