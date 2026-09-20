#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON=${PYTHON:-python}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}
CKPT=${CKPT:-$CHECKPOINT_DIR/FreezeEmpath}
DECODER=${DECODER:-$CHECKPOINT_DIR/indextts2_codes2wav}
WHISPER_MODEL=${WHISPER_MODEL:-$CHECKPOINT_DIR/whisper/large-v3.pt}
QUESTIONS=${1:-examples/manifest.jsonl}
OUTDIR=${2:-outputs/empathetic_dialogue}

if [[ ! -f "$WHISPER_MODEL" ]]; then
    echo "Missing Whisper checkpoint: $WHISPER_MODEL" >&2
    echo "Run: $PYTHON scripts/download_models.py --output_dir \"$CHECKPOINT_DIR\"" >&2
    exit 1
fi
mkdir -p "$OUTDIR"
"$PYTHON" inference/infer.py \
    --model_path "$CKPT" --speech_encoder "$WHISPER_MODEL" \
    --question_file "$QUESTIONS" --answer_file "$OUTDIR/answers.jsonl" \
    --temperature "${TEMPERATURE:-0.7}" --seed "${SEED:-42}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-256}" --s2s

"$PYTHON" inference/codes2wav.py \
    --cfg_path "$DECODER/config.yaml" --model_dir "$DECODER" \
    --input_file "$OUTDIR/answers.jsonl" --output_dir "$OUTDIR/wav" \
    --seed "${SEED:-42}"
