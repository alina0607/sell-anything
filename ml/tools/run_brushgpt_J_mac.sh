#!/bin/zsh
# Brush GPT run J on the Mac (owner, 2026-10-16: start the new cat-painting transformer as soon as the CAT painting is
# done). J = run I's recipe (aligned faces, face points first, eye-region loss x3, d512 x 7 layers) on the AFHQ cats
# plus the CAT-dataset cats, so the only difference from I is the data (paper-ideas P3). It is also the full-size
# baseline (5 tokens a stroke) that a later, compressed version will be compared with. One GPU job at a time:
#
#   1. wait for the painting loop (tools/run_mac_paint.sh) to finish, then paint the last stragglers of the list
#      (Colab has stopped and its paintings are merged, so nothing is painted twice);
#   2. tokens of both folders → tokens_J.npz, aligned with run H/I's eye midpoint and distance, and the stroke order
#      worked out again on the aligned faces (owner, 2026-10-16: CAT has upside-down and sideways cats whose order would
#      otherwise run bottom to top) → tokens_J_aligned_reordered.npz. The AFHQ cats get exactly H/I's strokes (checked),
#      in the re-worked order, so I is scored on its own file (tokens_runs_HI.npz) and J on its own: the same AFHQ
#      validation and test pictures, each model in its own stroke order;
#   3. train J (early stop on the validation loss, which now includes CAT cats). Batch 1 x accumulation 16: the same
#      16 rows (49,008 tokens) per optimiser step as I's 8 x 2, which ran out of MPS memory (18.8 GiB at batch 8,
#      still over 20 GiB at batch 4: on MPS the attention of a 3,063-token row is materialised). Everything else as
#      I: at most 8,000 steps (the cosine schedule depends on it), evaluation every 100 steps, patience 8. About 13 s
#      a step on the M5. The whole training state is checkpointed at every evaluation (last.pt); a training that stops
#      (watchdog, crash) is started again with --resume, up to 20 times, and continues from its last evaluation; the
#      weights are also kept every 1,000 steps (step_NNNNN.pt). caffeinate keeps the Mac awake while this runs. After a
#      power cut or a restart, start this script again: the data steps are skipped and the training resumes;
#   4. evaluate: brushgpt eval (256 samples, nearest-training-painting check), the eye judge for I and J at
#      temperatures 1.0, 0.8 and 0.6 (A1 Sketch-RNN notes: the paper also samples at low temperature), and the
#      losses split by source (tools/brushgpt_by_source.py: AFHQ val, AFHQ test, CAT val).
#
#     cd ml && (nohup tools/run_brushgpt_J_mac.sh >> runs/brushgpt/J_driver.log 2>&1 &)
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7 PYTHONPATH=.
NAME=J_cat_aligned_points_eyes
P=data/processed/brushgpt
DATA=$P/tokens_J_aligned_reordered.npz
R=runs/brushgpt/$NAME
PAINT=data/processed/cat_brush_face
echo $$ > runs/brushgpt/J_driver.pid
echo "J driver started $(date)"
caffeinate -ims -w $$ &

if [ ! -f $DATA ]; then
  while kill -0 "$(cat $PAINT/paint.pid)" 2>/dev/null; do sleep 60; done
  echo "painting loop (up to position 5,359) finished $(date)"
  tools/run_mac_paint.sh 7065 >> $PAINT/paint_run.log 2>&1
  echo "stragglers painted $(date): $(ls $PAINT/*.npz | wc -l) paintings"
  python3 -m sellanything_ml.sketch.brushgpt data --paint-dirs data/processed/afhq_brush_face $PAINT \
    --data $P/tokens_J.npz || { echo "data stage failed $(date)"; exit 1; }
  python3 -m sellanything_ml.sketch.brushgpt align --data $P/tokens_J.npz --aligned $DATA \
    --align-ref $P/tokens_runs_HI.npz --reorder-after 1 || { echo "align stage failed $(date)"; exit 1; }
  chmod a-w $P/tokens_J.npz $DATA
  shasum -a 256 $P/tokens_J.npz $DATA
fi
mkdir -p $R
if [ ! -f $R/by_source_I.json ]; then
  python3 tools/brushgpt_by_source.py --data $DATA --runs I_aligned_points_eyes --check $P/tokens_runs_HI.npz \
    --unordered --out $R/by_source_I.json || { echo "AFHQ strokes differ from runs H/I $(date)"; exit 1; }
  python3 tools/brushgpt_by_source.py --data $P/tokens_runs_HI.npz --runs I_aligned_points_eyes \
    --out $R/by_source_I_own_order.json
fi
echo "data ready $(date)"

for try in $(seq 1 20); do
  python3 -m sellanything_ml.sketch.brushgpt train --name $NAME --data $DATA \
    --d-model 512 --n-layers 7 --n-heads 8 --n-kv-heads 8 --dropout 0.1 --lr 3e-4 --batch 1 --accum 16 \
    --max-steps 8000 --warmup 100 --eval-every 100 --patience 8 --stop-after-min 2400 --keep-every 1000 \
    --order overlap --heads 1 --points 1 --eye-weight 3 --resume 1 >> $R/train.log 2>&1
  code=$?
  echo "J train try $try exit $code $(date)"
  [ $code -eq 0 ] && break
  sleep 120
done
[ -f $R/best.pt ] || { echo "no best.pt, stopping $(date)"; exit 1; }
shasum -a 256 $R/best.pt

python3 -m sellanything_ml.sketch.brushgpt eval --name $NAME --data $DATA --n 256 > $R/eval.log 2>&1
echo "J eval exit $? $(date)"
for run in $NAME I_aligned_points_eyes; do
  for t in 1.0 0.8 0.6; do
    python3 tools/eye_judge.py --run $run --n 128 --temperature $t --data $DATA \
      2>/dev/null | tail -1 >> $R/eye_judge.jsonl
    echo "eye judge $run t$t exit $? $(date)"
  done
done
python3 tools/brushgpt_by_source.py --data $DATA --runs $NAME \
  --out $R/by_source.json > $R/by_source.log 2>&1
echo "J finished $(date)"
