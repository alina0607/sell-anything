"""Brush GPT losses split by where the cats come from (AFHQ, the CAT dataset), on the validation and test splits.

A run trained on more cats has a different validation set, so its early-stopping loss does not compare with an older
run's. Here every run is scored on the same rows: the AFHQ validation cats and AFHQ's own test folder (never used for
early stopping by any run), plus the CAT-dataset validation cats. With `--data` an aligned file made with
`brushgpt.py align --align-ref tokens_runs_HI.npz`, the AFHQ rows are token for token those of runs H and I (checked
here against `--check`).

    PYTHONPATH=. python tools/brushgpt_by_source.py --data data/processed/brushgpt/tokens_J_aligned.npz \
        --runs I_aligned_points_eyes J_cat_aligned_points_eyes --check data/processed/brushgpt/tokens_runs_HI.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sellanything_ml.sketch import brushgpt as G


def source_of(ids: np.ndarray) -> np.ndarray:
    return np.where(np.char.startswith(ids.astype(str), "CAT_DATASET"), "cat", "afhq")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--check", type=Path, default=None, help="an older file whose AFHQ rows must be the same")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    z = np.load(a.data)
    if a.check is not None:
        old = np.load(a.check)
        where = {str(i): k for k, i in enumerate(z["ids"])}
        at = np.array([where[str(i)] for i in old["ids"]])
        same = {k: bool((z[k][at] == old[k]).all()) for k in old.files if k.startswith(("tokens", "points"))}
        print(json.dumps({"same_as_check": same}), flush=True)
        if not all(same.values()):
            raise SystemExit("the AFHQ rows differ from the check file")
    dev = G.device()
    report = {"data": str(a.data), "runs": {}}
    for run in a.runs:
        model = G.load_model(G.RUN_DIR / run / "best.pt", dev)
        cfg = model.brush_cfg
        mask = G.allowed(dev, cfg.points) if cfg.heads else None
        out = {}
        for split in ("val", "test"):
            rows, ids = G.load_rows(a.data, split, 0, cfg.order, points=cfg.points)
            pts = G.load_points(a.data, split)
            src = source_of(ids)
            for s in ("afhq", "cat"):
                keep = src == s
                if keep.any():
                    loss = G.slot_losses(model, rows[keep], dev, mask=mask, eyes=G.eye_strokes(rows[keep], pts[keep]))
                    out[f"{split}_{s}"] = {"cats": int(keep.sum()), "all": round(loss["all"], 4),
                                           "eye": round(loss["eye"], 4)}
        report["runs"][run] = out
        print(json.dumps({run: out}), flush=True)
    if a.out is not None:
        a.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
