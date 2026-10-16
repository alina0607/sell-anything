#!/bin/zsh
# Paint the CAT dataset's train cats on the Mac, from the front of the list (--n 7065 = the whole list; 4,065 until 2026-10-14), restarting after a slowdown stop (paint_cat_dataset.py
# stops itself when painting slows 2.5x). Started detached so it outlives any Claude session:
#
#     cd ml && (nohup tools/run_mac_paint.sh [N] > data/processed/cat_brush_face/paint_run.log 2>&1 &)
#
# N (default 7065) is where the Mac stops in the list. While Colab is painting down from the back, give the Mac a stop
# short of Colab so the two never paint the same cat (2026-10-15: Colab at 5,359 going down, Mac at 4,257 going up,
# Colab about twice as fast, so the Mac stops at 4,630 and Colab should be stopped after it paints position 4,630,
# CAT_DATASET_01/CAT_00/00000066_029.jpg). Colab then stopped for good at 5,359 (CAT_DATASET_02/CAT_04/00000926_020.jpg),
# so on 2026-10-16 the Mac was restarted with N = 5359 to paint everything before it; 5,359 to 5,430 are on Drive.
#
# Check:  cat data/processed/cat_brush_face/paint.pid; pgrep -fl paint_cat_dataset; ls data/processed/cat_brush_face/*.npz | wc -l
# Stop:   kill $(cat data/processed/cat_brush_face/paint.pid); pkill -f "paint_cat_dataset.py data/processed/cat_brush_face"
# (Colab paints the last 2,973 from the back; since 2026-10-15 the Mac also works through that part from its front, because
# Colab often runs out of free quota. Cats already painted are skipped, so nothing is painted twice on the Mac;
# tools/merge_colab_paint.py leaves a cat alone that the Mac already painted. See tools/prepare_colab_paint.py.)
cd "$(dirname "$0")/.." || exit 1
echo $$ > data/processed/cat_brush_face/paint.pid
N=${1:-7065}
for try in $(seq 1 30); do
    PYTHONPATH=. python3 tools/paint_cat_dataset.py data/processed/cat_brush_face --n $N
    code=$?
    echo "paint try $try exit $code $(date)"
    [ $code -eq 0 ] && break
    sleep 120
done
echo "painting loop finished $(date)"
