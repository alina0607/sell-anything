"""Record one autoplayed round as the README's GIF (docs/images/round.gif).

    python scripts/record_round.py                  # real models: classifier + buyer GPT
    python scripts/record_round.py --phrases-only   # real classifier + deterministic buyer phrases

Starts the ML server, runs the game with the OpenGL backend and --autoplay, captures every 5th frame
(12 per second at 60 fps, so the GIF plays in real time), and writes an 800×450 GIF. The window has to
stay visible: macOS slows a hidden window's frames to a crawl.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_autoplay import find_game, free_port, wait_for_server  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEP = 5                 # capture every 5th frame
MAX_FRAMES = 900         # autoplay ends the round well before this
SIZE = (800, 450)


def build_gif(frames: list[Image.Image], out: Path) -> None:
    # One palette for every frame, taken from a mosaic of frames spread over the round, so the small
    # buyer sprites' colors are in it. Max-coverage at 256 colors measured the lowest error on the
    # buyer cards (mean 0.91 per channel, against 2.88 for median cut, which turned red shirts brown).
    picks = frames[:: max(1, len(frames) // 12)][:12]
    mosaic = Image.new("RGB", (SIZE[0] * 4, SIZE[1] * 3))
    for i, frame in enumerate(picks):
        mosaic.paste(frame, ((i % 4) * SIZE[0], (i // 4) * SIZE[1]))
    palette = mosaic.quantize(colors=256, method=Image.Quantize.MAXCOVERAGE)
    q = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    durations = [round(1000 * STEP / 60)] * (len(q) - 1) + [2500]   # hold the sale at the end
    q[0].save(out, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phrases-only", action="store_true",
                        help="real sketch classifier with deterministic phrase-pool buyers")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "images" / "round.gif")
    args = parser.parse_args()

    port = free_port()
    extra = ["--phrases-only"] if args.phrases_only else []
    server = subprocess.Popen([sys.executable, "-m", "sellanything_ml.server", "--port", str(port), *extra],
                              cwd=ROOT / "ml", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            wait_for_server(port, server)
            captures = ",".join(f"{tmp}/f{f:04d}.png@{f}" for f in range(STEP, MAX_FRAMES, STEP))
            started = time.time()
            subprocess.run([str(find_game()), "--backend", "opengl", "--autoplay", "--server", f"127.0.0.1:{port}"],
                           env={**os.environ, "SA_CAPTURE": captures}, check=True, capture_output=True)
        finally:
            server.terminate()
            server.wait(timeout=10)
        paths = sorted(Path(tmp).glob("f*.png"))
        frames = [Image.open(p).convert("RGB").resize(SIZE, Image.LANCZOS) for p in paths]
    if not frames:
        print("no frames were captured")
        return 1
    build_gif(frames, args.out)
    destination = args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out
    print(f"{len(frames)} frames in {time.time() - started:.0f}s → {destination} "
          f"({args.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
