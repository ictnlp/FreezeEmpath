"""Download the inference checkpoints, or check an existing local installation."""
import argparse
import json
from pathlib import Path

MODEL_REPO = "ICTNLP/FreezeEmpath"
DECODER_REPO = "ICTNLP/indextts2_decoder"
MODEL_REVISION = "abeaefaeb36a6364837f832ef8fe4b9e10f99f17"
DECODER_REVISION = "2c346af2fdab3338cd13418c96e10f68cf28fb14"


def check_files(root):
    model = root / "FreezeEmpath"
    decoder = root / "indextts2_codes2wav"
    needed = [
        model / name for name in (
            "config.json", "model.safetensors.index.json", "tokenizer_config.json",
            "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json",
            "tts_tokenizer/tokenizer_config.json", "tts_tokenizer/vocab.json",
            "tts_tokenizer/merges.txt", "tts_tokenizer/added_tokens.json",
            "tts_tokenizer/special_tokens_map.json",
        )
    ]
    needed += [decoder / name for name in (
        "config.yaml", "semantic_codec.safetensors", "s2mel.pth", "bigvgan_generator.pt",
        "style.pt", "prompt_condition.pt", "ref_mel.pt",
    )]
    needed += [root / "whisper" / "large-v3.pt"]
    index = model / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        needed += [model / name for name in sorted(set(weight_map.values()))]
    missing = [str(path) for path in needed if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit("Missing or empty checkpoint files:\n" + "\n".join(missing))
    print(f"Found all {len(needed)} required files in {root}. This checks file presence, not inference or checksums.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--decoder_revision", default=DECODER_REVISION)
    parser.add_argument("--check", action="store_true", help="Check local file presence without network access")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if not args.check:
        from huggingface_hub import snapshot_download
        import whisper
        for repo, revision, directory in (
            (MODEL_REPO, args.model_revision, "FreezeEmpath"),
            (DECODER_REPO, args.decoder_revision, "indextts2_codes2wav"),
        ):
            print(f"Downloading {repo}@{revision} to {root / directory}", flush=True)
            snapshot_download(repo_id=repo, revision=revision, local_dir=str(root / directory))
        # Whisper verifies the official checkpoint checksum when downloading.
        model = whisper.load_model("large-v3", device="cpu", download_root=str(root / "whisper"))
        del model
    check_files(root)


if __name__ == "__main__":
    main()
