"""Paint CAT dataset cats (Zhang et al. 2008, 9 hand-labelled points per cat) the same way tools/paint_afhq.py paints
AFHQ cats: a square around the face (the points' box and 20% more), painted in the first four stages and snapped to
the token grid, one .npz of stages per cat, plus the face points in the painting's own pixels. The photos and subject
masks are the ones catface.py prepared; the cats are the catface train split, in its order. Finished cats are
skipped, so the run can be stopped and restarted: each cat is written (atomically) the moment it is finished.

How well the eyes came out is measured for every cat and logged (owner, 2026-10-12: drop the cats whose eyes were
not painted well; the threshold is set later from the distribution): `eye_l1`, the painting's error against the
photo inside the eye discs; `eye_conf` and `eye_conf_photo`, the landmark model's eye confidence on the painting
and on the photo it copies (tools/eye_judge.py); `eye_err`, how far the model finds the painting's eyes from the
hand-labelled ones, in eye-to-eye distances.

    PYTHONPATH=. python tools/paint_cat_dataset.py <out_dir> --n 20
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
import torch

from sellanything_ml.sketch import brushtokens, catbrush

ML = Path(__file__).resolve().parents[1]
RAW = ML / "data" / "raw" / "cat_dataset"
SPLIT = ML / "data" / "processed" / "catface" / "train.npz"
STAGES = 4
MARGIN = 1.2                      # the face square's side, in landmark-box sides (as for AFHQ)
WATCH_WINDOW, BASELINE_CATS, SLOWDOWN_FACTOR = 12, 12, 2.5
MIN_EYE_MASK = 0.5                # a cat whose subject mask is below this at either eye is skipped (the face would
                                  # be painted over in white: 1 of the first 20, 6 of the first 400)


def face_box(points: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """A square around the points, moved (and if need be shrunk) to lie inside a width × height photo."""
    centre = (points.min(0) + points.max(0)) / 2
    side = min(np.ptp(points, 0).max() * MARGIN, width, height)
    left = float(np.clip(centre[0] - side / 2, 0, width - side))
    top = float(np.clip(centre[1] - side / 2, 0, height - side))
    return round(left), round(top), round(left + side), round(top + side)


def eyes_in_mask(points: np.ndarray, mask_path: Path, size: tuple[int, int],
                 box: tuple[int, int, int, int]) -> bool:
    """Does the subject mask cover both eyes (points 0 and 1)? Read where catbrush.load reads it."""
    from PIL import Image
    mask = np.asarray(Image.open(mask_path).convert("L").resize(size).crop(box), np.float32) / 255
    at = ((points[:2] - box[:2]) / (box[2] - box[0]) * mask.shape[0]).astype(int).clip(0, mask.shape[0] - 1)
    return bool(min(mask[y, x] for x, y in at) >= MIN_EYE_MASK)


EYE_DISC = 0.35                   # eye-region radius in eye distances (brushgpt.EYE_ZONE)


def eye_checks(painting: np.ndarray, photo: np.ndarray, points: np.ndarray) -> dict[str, float]:
    """How well the painting shows the eyes, against the photo and the labelled eye points (canvas pixels)."""
    import eye_judge
    eyes = points[:2]
    dist = float(np.linalg.norm(eyes[0] - eyes[1]))
    yy, xx = np.mgrid[:painting.shape[0], :painting.shape[1]] + 0.5
    disc = np.zeros(painting.shape[:2], bool)
    for x, y in eyes:
        disc |= np.hypot(xx - x, yy - y) < EYE_DISC * dist
    found, conf = eye_judge.judge(np.stack([painting, photo]).astype(np.float32))
    return {"eye_l1": round(float(np.abs(painting - photo)[disc].mean()), 4),
            "eye_conf": round(float(conf[0, :2].mean()), 4), "eye_conf_photo": round(float(conf[1, :2].mean()), 4),
            "eye_err": round(float(np.linalg.norm(found[0, :2] - eyes, axis=1).mean() / dist), 4)}


def save_atomic(path: Path, **arrays: np.ndarray) -> None:
    """Write an .npz so that a crash never leaves half a file under the final name."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def points_of(full_id: str, zips: dict[str, zipfile.ZipFile]) -> np.ndarray:
    zstem, member = full_id.split("/", 1)
    z = zips.setdefault(zstem, zipfile.ZipFile(RAW / f"{zstem}.zip"))
    v = [float(x) for x in z.read(member + ".cat").decode().split()]
    return np.array(v[1:1 + 2 * int(v[0])]).reshape(-1, 2)


def main() -> None:
    """Paint the first --n cats of the catface train split that are not painted yet."""
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    from PIL import Image
    args.out.mkdir(parents=True, exist_ok=True)
    ids = [str(x) for x in np.load(SPLIT)["ids"][:args.n]]
    dev = "cuda" if torch.cuda.is_available() else "mps"
    zips: dict[str, zipfile.ZipFile] = {}
    baseline: list[float] = []
    recent: deque[float] = deque(maxlen=WATCH_WINDOW)
    log = args.out / "log.jsonl"
    skipped = {json.loads(line)["id"] for line in log.read_text().splitlines() if '"skipped"' in line} \
        if log.exists() else set()
    for full_id in ids:
        zstem, member = full_id.split("/", 1)
        name = f"{zstem}__{member.replace('/', '__')}"
        stem = Path(name).stem
        if (args.out / f"{stem}.npz").exists() or full_id in skipped:
            continue
        t0 = time.time()
        photo_path, mask_path = RAW / "photos" / name, RAW / "masks" / f"{stem}_mask1.png"
        if not mask_path.exists():
            continue
        points = points_of(full_id, zips)
        w, h = Image.open(photo_path).size
        box = face_box(points, w, h)
        if not eyes_in_mask(points, mask_path, (w, h), box):
            with open(args.out / "log.jsonl", "a") as f:
                f.write(json.dumps({"id": full_id, "skipped": "eyes outside the subject mask"}) + "\n")
            continue
        photo = catbrush.load(str(photo_path), str(mask_path), box)
        settings = catbrush.PaintSettings(dev=dev, snap=brushtokens.snapper(), log=lambda _: None, stages=STAGES)
        stages, frames = catbrush.paint(photo, settings)
        side = box[2] - box[0]
        canvas_points = (points - box[:2]) / side * catbrush.SIDE
        save_atomic(args.out / f"{stem}.npz", box=np.array(box), points=canvas_points,
                    **{f"s{k}_{key}": v for k, st in enumerate(stages) for key, v in st.items()})
        l1 = float(np.abs(frames[-1] - photo.target.numpy()).mean())
        checks = eye_checks(np.clip(frames[-1], 0, 1), photo.target.numpy(), canvas_points)
        del stages, frames, photo
        gc.collect()
        if dev == "mps":
            torch.mps.empty_cache()
        seconds = time.time() - t0
        recent.append(seconds)
        if len(baseline) < BASELINE_CATS:
            baseline.append(seconds)
        with open(args.out / "log.jsonl", "a") as f:
            f.write(json.dumps({"id": full_id, "box": box, "side": side, "l1": round(l1, 4),
                                "seconds": round(seconds, 1), **checks}) + "\n")
        if (len(baseline) == BASELINE_CATS and len(recent) == WATCH_WINDOW
                and statistics.median(recent) > statistics.median(baseline) * SLOWDOWN_FACTOR):
            raise RuntimeError("painting slowdown; stopped before wasting GPU time")


if __name__ == "__main__":
    main()
