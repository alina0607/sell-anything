"""An independent judge of whether paintings show a cat's eyes: the cat-face landmark model (cat-face-landmarks repo,
runs/pose_mae, trained on photos, never on paintings) finds the nine face points and gives each a confidence. A
painting that shows eyes should get eye confidences like real cats' paintings do.

    PYTHONPATH=. python tools/eye_judge.py --run D_all_cats --n 32
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np

LANDMARKS = Path.home() / "Digipen fall 2026" / "cat-face-landmarks"
PAD = 1.5                         # as when the AFHQ cats were labelled
EYES = [0, 1]


@functools.cache
def _model(dev: str):
    """The landmark model, loaded once per device."""
    sys.path.insert(0, str(LANDMARKS))
    from catlandmarks.finetune import load
    return load(LANDMARKS / "runs" / "pose_mae", dev)


def judge(images: np.ndarray, dev: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """(n, H, W, 3) images in 0–1 → points (n, 9, 2) in the images' pixels and confidences (n, 9)."""
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(LANDMARKS))
    from catlandmarks.pretrain import MEAN, STD
    model = _model(dev)
    size = model.encoder.cfg.image_size
    points, conf = [], []
    with torch.no_grad():
        for i in range(0, len(images), 32):
            x = torch.from_numpy(images[i:i + 32]).permute(0, 3, 1, 2).float().to(dev)
            x = F.interpolate(x, size=size, mode="bilinear", antialias=True)
            x = (x - MEAN.to(dev)) / STD.to(dev)
            inner = round(size / PAD)
            framed = torch.zeros_like(x)
            at = (size - inner) // 2
            framed[:, :, at:at + inner, at:at + inner] = F.interpolate(x, size=inner, mode="bilinear", antialias=True)
            p, c = model(framed)
            points.append(((p + 0.5 - at) * (images.shape[1] / inner) - 0.5).cpu().numpy())
            conf.append(c.cpu().numpy())
    return np.concatenate(points), np.concatenate(conf)


def eye_score(images: np.ndarray, dev: str = "cpu") -> dict[str, float]:
    """The judge's mean eye confidence over the images (with its standard error) and its mean for the other points."""
    _, c = judge(images, dev)
    eye = c[:, EYES].mean(1)
    return {"eye_conf_mean": round(float(eye.mean()), 4), "eye_conf_se": round(float(eye.std() / np.sqrt(len(eye))), 4),
            "other_conf_mean": round(float(c[:, 2:].mean()), 4)}


def main() -> None:
    """Print the judge's eye and other-point confidence for real validation paintings and a run's samples."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="D_all_cats")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--data", type=Path, default=None, help="the run's token file (default: brushgpt.DATA_FILE)")
    a = ap.parse_args()
    from sellanything_ml.sketch import brushgpt as G
    val, _ = G.load_rows(a.data or G.DATA_FILE, "val")
    real = np.stack([G.render(r) for r in val[:a.n]])
    model = G.load_model(G.RUN_DIR / a.run / "best.pt", G.device())
    gen = np.stack([G.render(r) for r in G.sample(model, a.n, a.temperature, seed=3, batch=min(a.n, 64))])
    print(json.dumps({"real_paintings": eye_score(real), f"{a.run}_t{a.temperature}": eye_score(gen)}))


if __name__ == "__main__":
    main()
