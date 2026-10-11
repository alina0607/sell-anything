"""Record the real game while scripted input holds Right; used before handing off character art."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from e2e_autoplay import find_game

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/tmp/girl-gameplay-run.gif")
FIRST, LAST, STEP = 45, 285, 6


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        captures = ",".join(f"{tmp}/f{frame:04d}.png@{frame}" for frame in range(FIRST, LAST + 1, STEP))
        env = {**os.environ, "SA_KEYS": "right@30:300", "SA_CAPTURE": captures, "SA_EXIT_AFTER": "310"}
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
