"""Fail when an idle character's visible body is not the standing height recorded in its sheet."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "game" / "assets" / "stage"
NAMES = ["girl_idle", *(f"npc_{i}_idle" for i in range(1, 8))]


def visible_box(image: Image.Image, threshold: int = 20) -> tuple[int, int, int, int]:
    alpha = image.getchannel("A").point(lambda value: 255 if value > threshold else 0)
    box = alpha.getbbox()
    if box is None:
        raise AssertionError("sprite has no visible pixels")
    return box


def main() -> None:
    failures: list[str] = []
    for name in NAMES:
        meta = json.loads((STAGE / f"{name}.json").read_text())
        image = Image.open(STAGE / f"{name}.png").convert("RGBA")
        width, height = meta["cell"]
        left, top, right, bottom = visible_box(image.crop((0, 0, width, height)))
        visible_height = bottom - top
        target = int(meta["standing"])
        delta = abs(visible_height - target)
        print(f"{name:12} visible={visible_height:3}px target={target:3}px width={right-left:3}px")
        if delta > 3:
            failures.append(f"{name}: visible height {visible_height}px != {target}px")
    if failures:
        raise SystemExit("character scale audit failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
