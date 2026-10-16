#!/bin/zsh
# Paint the CAT dataset's train cats on the Mac, from the front of the list (--n 7065 = the whole list; 4,065 until 2026-10-14), restarting after a slowdown stop (paint_cat_dataset.py
# stops itself when painting slows 2.5x). Started detached so it outlives any Claude session:
#
#     cd ml && (nohup tools/run_mac_paint.sh > data/processed/cat_brush_face/paint_run.log 2>&1 &)
#
# Check:  cat data/processed/cat_brush_face/paint.pid; pgrep -fl paint_cat_dataset; ls data/processed/cat_brush_face/*.npz | wc -l
# Stop:   kill $(cat data/processed/cat_brush_face/paint.pid); pkill -f "paint_cat_dataset.py data/processed/cat_brush_face"
# (Colab paints the last 2,973 from the back; since 2026-10-15 the Mac also works through that part from its front, because
# Colab often runs out of free quota. Cats already painted are skipped, so nothing is painted twice on the Mac;
# tools/merge_colab_paint.py leaves a cat alone that the Mac already painted. See tools/prepare_colab_paint.py.)
cd "$(dirname "$0")/.." || exit 1
echo $$ > data/processed/cat_brush_face/paint.pid
for try in $(seq 1 30); do
    PYTHONPATH=. python3 tools/paint_cat_dataset.py data/processed/cat_brush_face --n 7065
    code=$?
    echo "paint try $try exit $code $(date)"
    [ $code -eq 0 ] && break
    sleep 120
done
echo "painting loop finished $(date)"
