CKPT=checkpoints/FreezeEmpath

OUTDIR=$CKPT/answers/empathetic_dialogue
mkdir -p $OUTDIR

# python inference/infer.py \
#     --model_path $CKPT \
#     --question_file examples/manifest.jsonl \
#     --answer_file $OUTDIR/answers.jsonl \
#     --temperature 0.7 \
#     --input_type mel \
#     --mel_size 128 \
#     --s2s

python inference/codes2wav.py \
    --cfg_path checkpoints/indextts2_codes2wav/config.yaml \
    --model_dir checkpoints/indextts2_codes2wav \
    --input_file $OUTDIR/answers.jsonl \
    --output_dir $OUTDIR/wav 