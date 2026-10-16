"""Brush GPT experiments on one Modal A100 when the free Kaggle quota is used up (devlog 2026-10-11).

    cd ml && modal run tools/modal_brushgpt.py

Three runs on the same data and validation cats, one after another in one container (one GPU job at a time):
A the trial's setup on all painted cats, B + the "overlap" stroke order, C + one output layer per field. Each run
trains (early stop on the validation loss, a time limit) and is evaluated; the metrics, evaluations and images come
back to runs/brushgpt/<name>/, and run C's weights too. No Modal Volume is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

ML = Path(__file__).resolve().parents[1]
DATA = "data/processed/brushgpt/tokens.npz"
ALIGNED = "data/processed/brushgpt/tokens_aligned.npz"         # brushgpt.py align: faces with the eyes level
MODEL = ["--d-model", "512", "--n-layers", "7", "--n-heads", "8", "--n-kv-heads", "8", "--dropout", "0.1"]
TRAIN = ["--max-steps", "3000", "--warmup", "100", "--eval-every", "100", "--patience", "6", "--batch", "8",
         "--accum", "2"]
RUNS = {"A_list": ["--order", "list", "--heads", "0"],
        "B_overlap": ["--order", "overlap", "--heads", "0"],
        "C_overlap_heads": ["--order", "overlap", "--heads", "1"],
        # all 4,085 painted cats (the trial had 1,792), same setup as C, longer: later flags win over TRAIN's
        "D_all_cats": ["--order", "overlap", "--heads", "1", "--max-steps", "8000", "--patience", "8"],
        # devlog 2026-10-11 §8: D plus the face points first (E), plus eye-region weight 3 (F), plus both (G)
        "E_points": ["--order", "overlap", "--heads", "1", "--max-steps", "8000", "--patience", "8", "--points", "1"],
        "F_eye_weight": ["--order", "overlap", "--heads", "1", "--max-steps", "8000", "--patience", "8",
                         "--eye-weight", "3"],
        "G_points_eyes": ["--order", "overlap", "--heads", "1", "--max-steps", "8000", "--patience", "8",
                          "--points", "1", "--eye-weight", "3"],
        # devlog 2026-10-11 §9: D on the aligned faces (the shape GPT's alignment)
        "H_aligned": ["--order", "overlap", "--heads", "1", "--max-steps", "8000", "--patience", "8"],
        # H plus G's face points first and eye weight 3
        "I_aligned_points_eyes": ["--order", "overlap", "--heads", "1", "--max-steps", "8000", "--patience", "8",
                                  "--points", "1", "--eye-weight", "3"]}
# Each finished run is tied to the frozen, read-only copy of the tokens it was trained on (docs/model-versions.md), so
# it can be run again after tokens.npz is rebuilt with more cats. New runs not listed here use DATA.
TRIAL = "data/processed/brushgpt/tokens_trial_1792.npz"    # run C (1,639 train cats)
DEFG = "data/processed/brushgpt/tokens_runs_DEFG.npz"      # runs D-G (3,413 train cats; D ignores the face points)
HI = "data/processed/brushgpt/tokens_runs_HI.npz"          # runs H, I (the same cats, aligned)
DATA_OF = {"C_overlap_heads": TRIAL, "D_all_cats": DEFG, "E_points": DEFG, "F_eye_weight": DEFG,
           "G_points_eyes": DEFG, "H_aligned": HI, "I_aligned_points_eyes": HI}
KEEP_WEIGHTS = ("C_overlap_heads", "D_all_cats", "E_points", "F_eye_weight", "G_points_eyes", "H_aligned",
                "I_aligned_points_eyes")
SAMPLES = 256

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "numpy", "pillow", "scipy", "pytorch-fid==0.3.0")
         .add_local_dir(ML / "sellanything_ml", "/root/ml/sellanything_ml", ignore=["**/__pycache__/**"])
         .add_local_file(ML / DATA, f"/root/ml/{DATA}")
         .add_local_file(ML / ALIGNED, f"/root/ml/{ALIGNED}"))
for _frozen in (TRIAL, DEFG, HI):
    image = image.add_local_file(ML / _frozen, f"/root/ml/{_frozen}")
app = modal.App("brushgpt-runs", image=image)


@app.function(gpu="A100-40GB", timeout=4 * 3600)
def run_all(names: list[str], minutes: float) -> dict[str, bytes]:
    import subprocess
    files: dict[str, bytes] = {}
    for name in names:
        extra = RUNS[name] + ["--stop-after-min", str(minutes)]
        base = ["python", "-m", "sellanything_ml.sketch.brushgpt"]
        common = ["--name", name, "--data", DATA_OF.get(name, DATA), "--runs", "/root/runs"]
        for args in (base + ["train"] + common + MODEL + TRAIN + extra, base + ["eval"] + common + ["--n", str(SAMPLES)]):
            proc = subprocess.run(args, cwd="/root/ml", capture_output=True, text=True)
            files[f"{name}/{args[3]}.log"] = (proc.stdout + proc.stderr).encode()
            if proc.returncode != 0:
                break
        for p in Path("/root/runs", name).glob("*"):
            if p.suffix in (".json", ".jsonl", ".png", ".txt") or (p.name == "best.pt" and name in KEEP_WEIGHTS):
                files[f"{name}/{p.name}"] = p.read_bytes()
    return files


@app.local_entrypoint()
def main(runs: str = ",".join(RUNS), minutes: float = 35) -> None:
    """--runs C_overlap_heads --minutes 15 runs only C with a 15-minute training limit."""
    names = runs.split(",")
    out = ML / "runs" / "brushgpt"
    for rel, data in run_all.remote(names, minutes).items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(data)
    for name in names:
        ev = out / name / "eval.json"
        print(name, ev.read_text() if ev.exists() else "(no eval: see the logs)")
    print(json.dumps({"saved_to": str(out)}))
