"""Record real-game right/left runs, away/toward walks and a jump before handing off character art."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from e2e_autoplay import find_game

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/tmp/girl-gameplay-movement-qa.gif")
FIRST, LAST, STEP = 30, 750, 6


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        captures = ",".join(f"{tmp}/f{frame:04d}.png@{frame}" for frame in range(FIRST, LAST + 1, STEP))
        keys = "right@30:180,left@210:360,up@390:480,down@510:600,space@630:632"
        env = {**os.environ, "SA_KEYS": keys, "SA_CAPTURE": captures, "SA_EXIT_AFTER": "760"}
        run = subprocess.run([str(find_game()), "--backend", "opengl"], cwd=ROOT, env=env,
                             capture_output=True, text=True)
        paths = sorted(Path(tmp).glob("f*.png"))
        if not paths:
            print(run.stdout)
            print(run.stderr)
            return 1
        frames = [Image.open(path).convert("RGB").resize((800, 450), Image.Resampling.LANCZOS) for path in paths]
        frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=round(1000 * STEP / 60),
                       disposal=2, loop=0)
    print(f"recorded {len(frames)} real-game frames -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
