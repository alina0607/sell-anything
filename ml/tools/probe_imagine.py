"""Why do a buyer model's scenes break the rules? Writes every imagine prompt of the two B2 validation sets, on the CPU,
the way tools/eval_buyer_b1.py samples (temperature 0.7, top-k 50, the prompt's index in its file as the seed), and
saves each line with its checks, so the failures can be counted and read.

    cd ml && python tools/probe_imagine.py runs/sft_b2/best.pt --limit 64 --out runs/sft_b2/imagine_probe/b2_64.jsonl
    cd ml && python tools/probe_imagine.py runs/sft_b2/best.pt --forced --out runs/sft_b2/imagine_probe/b2_forced.jsonl

`--limit` is the most tokens the model may write (the game and the evaluation allow MAX_NEW["imagine"] = 64), to
see whether longer scenes are being cut off. `--forced` starts the scene with the three things the written sale
picked ("a, b, c |"), so the model only writes the line: picking things and writing about them can be told apart.

The CPU is used so that the run can share the Mac with a GPU job. Random numbers differ between the CPU and MPS, so
these lines are not the same samples as the evaluation's; compare runs of this tool with each other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ML), str(ML / "tools")]
SETS = ("old", "new")


def main() -> None:
    import torch

    import eval_buyer_b1 as E
    from sellanything_ml.buyers.format import END
    from sellanything_ml.llm.evaluate import load_model
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--limit", type=int, default=64, help="most tokens the model may write")
    ap.add_argument("--forced", action="store_true", help="start with the written sale's three things")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    model, tok = load_model(a.checkpoint, torch.device("cpu"))
    rows = []
    for which in SETS:
        examples = [json.loads(line) for line in open(ML / f"runs/eval_sets_b2/val_{which}_format5.jsonl")]
        for i, ex in enumerate(examples):
            if ex["task"] != "imagine":
                continue
            prefix = ex["target"].split("|")[0] + "|" if a.forced else ""
            torch.manual_seed(i)
            ids = torch.tensor([tok.encode(ex["prompt"] + prefix)])
            with torch.no_grad():
                gen = model.generate(ids, a.limit, temperature=E.TEMPERATURE, top_k=E.TOP_K, stop_id=tok.eos_id)
            new = gen[0, ids.size(1):].tolist()
            text = (prefix + tok.decode(new).replace(END, "")).strip()
            rows.append({"set": which, "index": i, "seed": ex["seed"], "name": ex["name"], "sale": ex["target"],
                         "model": text, "tokens": len(new), "hit_limit": tok.eos_id not in new, **E.check(ex, text)})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{a.out}: {len(rows)} scenes, {sum(bool(r.get('rules_ok')) for r in rows) / len(rows):.1%} pass the rules")


if __name__ == "__main__":
    main()
