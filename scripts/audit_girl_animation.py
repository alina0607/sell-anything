"""Reject visible girl-animation regressions and render review artifacts.

Checks the built sheets, not source promises: four equal 500 ms run phases, locked head dimensions,
matched value/saturation, distinct opposing silhouettes, and consistent scale across idle/walk/run.
Writes /tmp/girl-animation-contact.png and /tmp/girl-run-audit.gif for human review.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "game/assets/stage"


def load(name: str) -> tuple[list[Image.Image], dict]:
    meta = json.loads((STAGE / f"{name}.json").read_text())
    sheet = Image.open(STAGE / f"{name}.png").convert("RGBA")
    w, h = meta["cell"]
    frames = []
    for i in range(meta["frames"]):
        x, y = (i % meta["cols"]) * w, (i // meta["cols"]) * h
        frames.append(sheet.crop((x, y, x + w, y + h)))
    return frames, meta


def metrics(frame: Image.Image) -> dict[str, float]:
    a = np.asarray(frame)
    alpha = a[..., 3]
    ys0, xs0 = np.nonzero(alpha > 20)
    y0, y1 = ys0.min(), ys0.max() + 1
    cut = y0 + round((y1 - y0) * 0.60)
    yy = np.indices(alpha.shape)[0]
    ys, xs = np.nonzero((alpha > 20) & (yy < cut))
    rgb = a[..., :3][alpha > 20].astype(float)
    return {
        "head_w": float(np.quantile(xs, .98) - np.quantile(xs, .02)),
        "head_h": float(np.quantile(ys, .98) - np.quantile(ys, .02)),
        "value": float(rgb.mean(1).mean()),
        "saturation": float((rgb.max(1) - rgb.min(1)).mean()),
    }


def silhouette_difference(a: Image.Image, b: Image.Image) -> float:
    aa = np.asarray(a.getchannel("A")) > 40
    bb = np.asarray(b.getchannel("A")) > 40
    return float(np.logical_xor(aa, bb).sum() / max(np.logical_or(aa, bb).sum(), 1))


def flatten(frame: Image.Image, size: tuple[int, int] = (366, 327)) -> Image.Image:
    canvas = Image.new("RGBA", size, "#fff4ec")
    canvas.alpha_composite(frame, ((size[0] - frame.width) // 2, size[1] - frame.height))
    return canvas.convert("RGB")


def main() -> int:
    idle, _ = load("girl_idle")
    walk, walk_meta = load("girl_walk")
    run, run_meta = load("girl_run")
    first, last = run_meta["parts"]["run"]
    cycle = run[first:last + 1]
    cycle_ms = run_meta["frame_ms"][first:last + 1]

    checks: list[tuple[bool, str]] = []
    checks.append((len(cycle) == 4 and cycle_ms == [500] * 4,
                   f"four equal run phases at 500 ms: {cycle_ms}"))
    cm = [metrics(f) for f in cycle]
    for key, limit in (("head_w", 8.0), ("head_h", 4.0), ("value", 2.0), ("saturation", 2.0)):
        spread = max(m[key] for m in cm) - min(m[key] for m in cm)
        checks.append((spread <= limit, f"run {key} spread {spread:.1f} <= {limit:.1f}"))

    refs = [metrics(idle[0]), metrics(walk[walk_meta["parts"]["away"][0]]),
            metrics(walk[walk_meta["parts"]["toward"][0]])]
    target_value = sum(m["value"] for m in refs) / len(refs)
    target_sat = sum(m["saturation"] for m in refs) / len(refs)
    run_value = sum(m["value"] for m in cm) / len(cm)
    run_sat = sum(m["saturation"] for m in cm) / len(cm)
    checks.append((abs(run_value - target_value) <= 3.0,
                   f"run/reference value delta {run_value - target_value:+.1f} <= 3.0"))
    checks.append((abs(run_sat - target_sat) <= 3.0,
                   f"run/reference saturation delta {run_sat - target_sat:+.1f} <= 3.0"))
    checks.append((silhouette_difference(cycle[0], cycle[2]) >= .10,
                   f"opposing stride silhouette difference {silhouette_difference(cycle[0], cycle[2]):.3f} >= 0.100"))
    checks.append((silhouette_difference(cycle[1], cycle[3]) >= .06,
                   f"crossing silhouette difference {silhouette_difference(cycle[1], cycle[3]):.3f} >= 0.060"))

    labels = ["idle", "away", "toward", "stride A", "cross A", "stride B", "cross B"]
    selected = [idle[0], walk[0], walk[walk_meta["parts"]["toward"][0]], *cycle]
    tiles = []
    for label, frame in zip(labels, selected, strict=True):
        tile = flatten(frame)
        ImageDraw.Draw(tile).text((8, 8), label, fill="#5b2f28")
        tiles.append(tile)
    contact = Image.new("RGB", (tiles[0].width * len(tiles), tiles[0].height), "#fff4ec")
    for i, tile in enumerate(tiles):
        contact.paste(tile, (i * tile.width, 0))
    contact.save("/tmp/girl-animation-contact.png")

    gif = [flatten(frame, run_meta["cell"]) for frame in cycle]
    gif[0].save("/tmp/girl-run-audit.gif", save_all=True, append_images=gif[1:], duration=cycle_ms,
                disposal=2, loop=0)

    for ok, message in checks:
        print(("PASS " if ok else "FAIL ") + message)
    print("cycle metrics:", [{k: round(v, 1) for k, v in m.items()} for m in cm])
    return 0 if all(ok for ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
