"""Buyer GPT B2 on one Modal A100 (owner, 2026-10-12: Modal credits left $4.01).

    cd ml && modal run tools/modal_sft_b2.py

Same recipe as B1 (pretrain_large, 3 epochs, lr 3e-4, per-token loss) on the B2 data (data/processed/buyers_b2: the
100 old sales plus the 150 sales of seeds 2200-2349, format 5, K = 2), batch 16. Cost is capped twice: sft.py stops
itself after MAX_MINUTES (evaluating and saving first), and the function's timeout ends the container. Returns
best.pt, the metrics and the log into runs/sft_b2/; the held-out evaluation runs on the Mac afterwards. No Volume.
"""

from __future__ import annotations

from pathlib import Path

import modal

ML = Path(__file__).resolve().parents[1]
INIT = "runs/pretrain_large/final.pt"
DATA = "data/processed/buyers_b2"
RUN = "runs/sft_b2"
MAX_MINUTES = 25                     # training time; the A100 bill also counts image start and the final copy

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "numpy", "sentencepiece", "pyarrow")
         .add_local_dir(ML / "sellanything_ml", "/ml/sellanything_ml", ignore=["**/__pycache__/**"])
         .add_local_file(ML / "checkpoints" / "tokenizer" / "sp8192.model", "/ml/checkpoints/tokenizer/sp8192.model")
         .add_local_dir(ML / DATA, f"/ml/{DATA}")
         .add_local_file(ML / INIT, f"/ml/{INIT}"))
for name in ["item_schema.json", "item_splits.json", "items.tsv", "props.json", "traits.json"]:
    image = image.add_local_file(ML / "data" / name, f"/ml/data/{name}")
app = modal.App("buyer-gpt-b2", image=image)


@app.function(gpu="A100-40GB", timeout=40 * 60)
def train() -> dict[str, bytes]:
    import subprocess
    args = ["python", "-m", "sellanything_ml.llm.sft", "--init", INIT, "--data", DATA, "--run-dir", RUN,
            "--epochs", "3", "--batch", "16", "--lr", "3e-4", "--eval-every", "500",
            "--max-minutes", str(MAX_MINUTES)]
    proc = subprocess.run(args, cwd="/ml", capture_output=True, text=True)
    files = {"train.log": (proc.stdout + proc.stderr).encode()}
    for name in ("best.pt", "metrics.jsonl"):
        path = Path("/ml", RUN, name)
        if path.exists():
            files[name] = path.read_bytes()
    return files


@app.local_entrypoint()
def main() -> None:
    out = ML / RUN
    out.mkdir(parents=True, exist_ok=True)
    for name, data in train.remote().items():
        (out / name).write_bytes(data)
    print((out / "train.log").read_text()[-1500:])
