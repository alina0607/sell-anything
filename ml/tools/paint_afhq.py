"""Paint every AFHQ cat's face with the straight brush, snapped to the token grid, one .npz of stages per cat.

Each 512-pixel photo is cut to a square around the face (the landmarks' box and 20% more), so the canvas holds the
face only, and painted in the first four stages: half the tokens of the whole frame in five (docs/devlog/2026-10-10.md).
Cats that pass the training-data rules (no ear tip more than 0.1 eye distance outside the image, mirror
disagreement outside the top 5%) come first; the rest are painted after them. Finished cats are skipped, so the
run can be stopped and restarted.

    python tools/paint_afhq.py <labels.npz> <labels_checks.npz> <out_dir>
"""

import gc
import json
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from sellanything_ml.sketch import brushtokens, catbrush

ML = Path(__file__).resolve().parents[1]
AFHQ, MASKS = ML / "data" / "raw" / "afhq", ML / "data" / "processed" / "afhq_masks"
STAGES = 4
MARGIN = 1.2                     # the face square's side, in landmark-box sides
SOURCE = 512
BASELINE_CATS = 24
WATCH_WINDOW = 12
SLOWDOWN_FACTOR = 2.5


def face_box(points: np.ndarray) -> tuple[int, int, int, int]:
    """A square around the landmarks, moved (and if need be shrunk) to lie inside the photo."""
    centre = (points.min(0) + points.max(0)) / 2
    side = min(np.ptp(points, 0).max() * MARGIN, SOURCE)
    left, top = np.clip(centre - side / 2, 0, SOURCE - side)
    return round(left), round(top), round(left + side), round(top + side)


def main() -> None:
    """Command line: see the module docstring."""
    labels, checks, out = np.load(sys.argv[1]), np.load(sys.argv[2]), Path(sys.argv[3])
    if not (labels["ids"] == checks["ids"]).all():
        raise ValueError("labels and checks list different cats")
    out.mkdir(parents=True, exist_ok=True)
    mirror = np.delete(checks["mirror"], [4, 7], axis=1).mean(1)      # ear tips left out: cut-off tips are guesses
    kept = (checks["tip_outside"] <= 0.1) & (mirror <= np.quantile(mirror, 0.95))
    order = np.r_[np.where(kept)[0], np.where(~kept)[0]]
    dev = "cuda" if torch.cuda.is_available() else "mps"
    baseline: list[float] = []
    recent: deque[float] = deque(maxlen=WATCH_WINDOW)
    for i in order:
        rel = str(labels["ids"][i])
        name = Path(rel).stem
        if (out / f"{name}.npz").exists():
            continue
        t0 = time.time()
        box = face_box(labels["points"][i])
        photo = catbrush.load(str(AFHQ / rel), str(MASKS / f"{name}_mask1.png"), box)
        settings = catbrush.PaintSettings(dev=dev, snap=brushtokens.snapper(), log=lambda _: None, stages=STAGES)
        stages, frames = catbrush.paint(photo, settings)
        np.savez(out / f"{name}.npz", box=np.array(box),
                 **{f"s{k}_{key}": v for k, st in enumerate(stages) for key, v in st.items()})
        l1 = float(np.abs(frames[-1] - photo.target.numpy()).mean())
        del stages, frames, photo
        gc.collect()
        if dev == "mps":
            torch.mps.empty_cache()   # without this the process grew to 9 GB and a cat took 10× as long
        seconds = time.time() - t0
        recent.append(seconds)
        if len(baseline) < BASELINE_CATS:
            baseline.append(seconds)
        baseline_median = statistics.median(baseline)
        recent_median = statistics.median(recent)
        memory = {}
        if dev == "mps":
            memory = {
                "mps_current_mb": round(torch.mps.current_allocated_memory() / 2**20, 1),
                "mps_driver_mb": round(torch.mps.driver_allocated_memory() / 2**20, 1),
            }
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps({"id": rel, "kept": bool(kept[i]), "box": box, "l1": round(l1, 4),
                                "seconds": round(seconds, 1), "recent_median": round(recent_median, 1),
                                "baseline_median": round(baseline_median, 1), **memory}) + "\n")
        if (len(baseline) == BASELINE_CATS and len(recent) == WATCH_WINDOW
                and recent_median > baseline_median * SLOWDOWN_FACTOR):
            raise RuntimeError(
                f"painting slowdown: recent median {recent_median:.1f}s is over "
                f"{SLOWDOWN_FACTOR:.1f}× the {baseline_median:.1f}s baseline; stopped before wasting GPU time"
            )


if __name__ == "__main__":
    main()
