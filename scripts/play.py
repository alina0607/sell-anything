"""Play Sell Anything with one command: starts a private ML server, then the game.

    python3 scripts/play.py --demo       # trained drawing recognition + deterministic phrase-pool buyers
    python3 scripts/play.py              # play with the installed trained models
    python3 scripts/play.py --autoplay   # watch it play a round (any game option passes through)

Works from any folder. A private server is started on the first free local port at or above 5555 (its
output goes to build/server.log) and stopped again when the game closes. Starting a private server is
intentional: an old test server may be configured to call every drawing "bicycle". The game is built
first if it hasn't been. `scripts/make_mac_app.py` wraps this in a double-clickable app.
"""

from __future__ import annotations

import subprocess
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_autoplay import find_game, wait_for_server  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PORT = 5555
LOG = ROOT / "build" / "server.log"


def free_port(start: int = PORT) -> int:
    """Choose our own port, so a stale --fixed-sketch test server can never affect play."""
    for port in range(start, start + 100):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No free local port found for the AI server")


def ml_python() -> Path:
    """Use the project's installed ML environment even when play.py is launched by system Python."""
    candidates = [ROOT / "ml" / ".venv" / "bin" / "python",
                  ROOT / "ml" / ".venv" / "Scripts" / "python.exe"]
    return next((path for path in candidates if path.exists()), Path(sys.executable))


def build_game() -> Path:
    try:
        return find_game()
    except FileNotFoundError:
        print("Building the game (first run only, a few minutes)...", flush=True)
        if not (ROOT / "build" / "CMakeCache.txt").exists():
            subprocess.run(["cmake", "-S", str(ROOT), "-B", str(ROOT / "build"), "-DCMAKE_BUILD_TYPE=Release"], check=True)
        subprocess.run(["cmake", "--build", str(ROOT / "build"), "--target", "sell_anything", "-j8"], check=True)
        return find_game()


def start_server(port: int, demo: bool = False) -> subprocess.Popen:
    print("Starting the demo server..." if demo else "Starting the AI server...", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG, "w")
    command = [str(ml_python()), "-m", "sellanything_ml.server", "--port", str(port)]
    if demo:
        # Demo keeps buyer dialogue deterministic, but still uses the real 345-class sketch model.
        command += ["--phrases-only"]
    server = subprocess.Popen(command, cwd=ROOT / "ml", stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_for_server(port, server, timeout=60)
    except RuntimeError as e:
        server.terminate()
        print(f"The AI server didn't start ({e}). Its log, {LOG}:\n{LOG.read_text()[-2000:]}")
        sys.exit(1)
    return server


def main() -> int:
    game_args = sys.argv[1:]
    demo = "--demo" in game_args
    game_args = [arg for arg in game_args if arg != "--demo"]
    game = build_game()
    port = free_port()
    server = start_server(port, demo)
    try:
        return subprocess.run([str(game), "--server", f"127.0.0.1:{port}", *game_args]).returncode
    except KeyboardInterrupt:
        return 130
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        time.sleep(0.1)


if __name__ == "__main__":
    sys.exit(main())
