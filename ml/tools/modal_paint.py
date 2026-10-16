"""Paint part of the AFHQ cats on Modal GPUs while the Mac paints the rest (owner, 2026-10-11: half on Modal, to finish
sooner). Same painting as tools/paint_afhq.py: same face square, same settings, one .npz of stages per cat, written
into the same folder as the Mac's. The Mac's run skips any cat whose .npz already exists, so the two never paint the
same cat as long as Modal takes the back of the list the Mac works through from the front.

    cd ml && modal run tools/modal_paint.py --test 3                      # speed check: three cats, nothing kept
    cd ml && modal run tools/modal_paint.py --share 0.5                   # the back half of the cats still to paint

Each call paints a few cats and sends their files back as soon as it ends, so a crash loses one chunk, not the run.
Photos and masks go up with each chunk; no Modal Volume is used.
"""

from __future__ import annotations

import io
import json
import statistics
import sys
import time
from pathlib import Path

import modal

ML = Path(__file__).resolve().parents[1]
LABELS = ML / "data" / "processed" / "afhq_cats_mae_pad1.5.npz"
CHECKS = ML / "data" / "processed" / "afhq_cats_mae_pad1.5_checks.npz"
OUT = ML / "data" / "processed" / "afhq_brush_face"
CHUNK = 8                 # cats per call
SLOWDOWN_FACTOR = 2.5     # stop when a chunk's median is this many times the first chunks' median
MAX_CONTAINERS = 4        # GPUs at once

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "numpy", "pillow")
         .add_local_dir(ML / "sellanything_ml", "/root/ml/sellanything_ml", ignore=["**/__pycache__/**"]))
app = modal.App("cat-painting", image=image)


@app.function(gpu="L4", timeout=40 * 60, max_containers=MAX_CONTAINERS)
def paint_chunk(cats: list[dict[str, object]]) -> list[dict[str, object]]:
    """Paint each cat (its photo and mask bytes, face box) and return its .npz bytes and seconds."""
    import tempfile

    import numpy as np
    import torch
    sys.path.insert(0, "/root/ml")
    from sellanything_ml.sketch import brushtokens, catbrush

    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for cat in cats:
            t0 = time.time()
            photo_path, mask_path = Path(tmp, "photo.png"), Path(tmp, "mask.png")
            photo_path.write_bytes(cat["photo"])
            mask_path.write_bytes(cat["mask"])
            box = tuple(cat["box"])
            photo = catbrush.load(str(photo_path), str(mask_path), box)
            settings = catbrush.PaintSettings(dev="cuda", snap=brushtokens.snapper(), log=lambda _: None, stages=4)
            stages, frames = catbrush.paint(photo, settings)
            buf = io.BytesIO()
            np.savez(buf, box=np.array(box), **{f"s{k}_{key}": v for k, st in enumerate(stages) for key, v in st.items()})
            l1 = float(np.abs(frames[-1] - photo.target.numpy()).mean())
            out.append({"id": cat["id"], "npz": buf.getvalue(), "l1": round(l1, 4),
                        "seconds": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name()})
    return out


def _still_to_paint() -> list[tuple[str, tuple[int, int, int, int]]]:
    """The kept cats (training-data rules) not painted yet, in the Mac's order, with their face squares."""
    import numpy as np
    sys.path.insert(0, str(ML / "tools"))
    from paint_afhq import face_box
    labels, checks = np.load(LABELS), np.load(CHECKS)
    mirror = np.delete(checks["mirror"], [4, 7], axis=1).mean(1)
    kept = (checks["tip_outside"] <= 0.1) & (mirror <= np.quantile(mirror, 0.95))
    return [(str(labels["ids"][i]), face_box(labels["points"][i])) for i in np.where(kept)[0]
            if not (OUT / f"{Path(str(labels['ids'][i])).stem}.npz").exists()]


def _payload(rel: str, box: tuple[int, int, int, int]) -> dict[str, object]:
    name = Path(rel).stem
    return {"id": rel, "box": list(box), "photo": (ML / "data" / "raw" / "afhq" / rel).read_bytes(),
            "mask": (ML / "data" / "processed" / "afhq_masks" / f"{name}_mask1.png").read_bytes()}


@app.local_entrypoint()
def main(test: int = 0, share: float = 0.5) -> None:
    """--test N paints N cats and reports speed (files not kept); otherwise paints the back `share` of the cats
    still to paint, MAX_CONTAINERS GPUs at a time, writing each chunk's files as it returns."""
    todo = _still_to_paint()
    if test:
        mine = todo[-test:]
        rows = paint_chunk.remote([_payload(rel, box) for rel, box in mine])
        print(json.dumps([{k: r[k] for k in ("id", "seconds", "l1", "gpu")} for r in rows], indent=1))
        return
    mine = todo[len(todo) - round(len(todo) * share):]
    print(f"{len(todo)} cats still to paint; Modal takes the last {len(mine)}")
    chunks = [mine[i:i + CHUNK] for i in range(0, len(mine), CHUNK)]
    baseline: list[float] = []
    done = 0
    inputs = ([_payload(rel, box) for rel, box in c] for c in chunks)       # read from disk only when sent
    for rows in paint_chunk.map(inputs, order_outputs=False):
        for r in rows:
            (OUT / f"{Path(r['id']).stem}.npz").write_bytes(r["npz"])
            with open(OUT / "log_modal.jsonl", "a") as f:
                f.write(json.dumps({k: r[k] for k in ("id", "seconds", "l1", "gpu")}) + "\n")
        done += len(rows)
        median = statistics.median(r["seconds"] for r in rows)
        if len(baseline) < 3:
            baseline.append(median)
        elif median > statistics.median(baseline) * SLOWDOWN_FACTOR:
            raise RuntimeError(f"painting slowdown: chunk median {median:.1f}s; stopped before wasting GPU time")
        print(f"{done}/{len(mine)} painted, chunk median {median:.1f}s", flush=True)
