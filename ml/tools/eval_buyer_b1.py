"""Held-out evaluation of the buyer GPT (B plan, prompt formats 4 and 5) on the validation sales.

    PYTHONPATH=. python tools/eval_buyer_b1.py runs/sft_b2/best.pt --out runs/sft_b2/eval.json
    PYTHONPATH=. python tools/eval_buyer_b1.py runs/sft_b1/best.pt --data data/processed/buyers_b1/val.jsonl --out …

Buyers have no spending limit (owner, 2026-10-12), so no budget is checked. An offer move counts as reasonable when it
is at most MAX_MOVE of the offer held and the offer stays within MAX_PEAK times the buyer's opening offer: in the
100 sales, 99% of raises are at most 33% and 99% of cuts at most 37%, and the highest offer of a buyer is at most
1.38 times their opening for 99% of buyers (1.62 at most). A question counts as a repeat when at least half of its
content words were in one earlier question (a rough measure of "the same meaning").

The model completes every validation prompt the way the game samples (temperature 0.7, top-k 50, a fixed seed per
prompt) and each line is checked by the same program rules the training sales passed, against the buyer's card in
the prompt; no other model judges. Also how often the model moves its offer the same way the sale's buyer did.
A readable copy of some examples goes next to the report (`*_samples.md`).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from sellanything_ml.buyers import stories
from sellanything_ml.buyers.cat_market import MAX_NEW, parse
from sellanything_ml.buyers.format import END
from sellanything_ml.buyers.sales_sft import OUT_DIR

SAID_AMOUNT = re.compile(r"\$(\d+)")             # a dollar amount in a spoken line
OFFER_NOW = re.compile(r"offer now \$(\d+)")
OPENING = re.compile(r"Your opening \(\$(\d+)\)")
TEMPERATURE, TOP_K = 0.7, 50                       # as the game samples (cat_market.BuyerGPT)
MAX_MOVE, MAX_PEAK, SAME_QUESTION = 0.40, 1.6, 0.5
TASKS = ("open", "ask", "react", "imagine")
RATES = ("spoken_ok", "new_question", "new_in_meaning", "rules_ok", "same_direction", "said_equals_offer", "reasonable")
SAMPLES_PER_TASK = 12


def context(ex: dict[str, Any]) -> dict[str, Any]:
    """Offer held, opening offer, the seller's latest answer and the questions already asked, read from the prompt."""
    now, opening = OFFER_NOW.search(ex["prompt"]), OPENING.search(ex["prompt"])
    lines = ex["prompt"].split("\n")
    seller = [line[len("Seller: "):] for line in lines if line.startswith("Seller: ")]
    asked = [q for line in lines if line.startswith("Asked earlier: ") for q in line[15:].split(" / ")]
    asked += [line.split(" asked: ", 1)[1] for line in lines if " asked: " in line]
    return {"offer": int(now[1]) if now else None, "opening": int(opening[1]) if opening else None,
            "answer": seller[-1] if seller else "", "asked": [q.strip().lower() for q in asked]}


def repeats(question: str, asked: list[str]) -> bool:
    """Does the question share at least SAME_QUESTION of its content words with one earlier question?"""
    words = stories._words(question)
    return bool(words) and any(len(words & stories._words(q)) >= SAME_QUESTION * len(words) for q in asked)


def _react_checks(row: dict[str, Any], ex: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    """A reaction: the program rules, the direction and size of the move against the sale's own buyer, the amount
    said against the offer, and whether the move is reasonable (MAX_MOVE, MAX_PEAK)."""
    problem = stories.reaction_problem(row, {}, c["offer"], ex["name"], c["answer"])
    want = parse("react", ex["target"].replace(END, ""))
    offer, held = row["offer"], c["offer"]
    said = set(map(int, SAID_AMOUNT.findall(row["says"])))
    return {"rules_ok": problem is None, "problem": problem,
            "same_direction": (offer > held) == (want["offer"] > held) and (offer < held) == (want["offer"] < held),
            "withdrew": offer == 0, "teacher_withdrew": want["offer"] == 0, "offer_error": abs(offer - want["offer"]),
            "said_equals_offer": not offer or not (said - {offer}),
            "reasonable": offer == 0 or (abs(offer - held) <= MAX_MOVE * held
                                         and offer <= MAX_PEAK * (c["opening"] or held))}


def check(ex: dict[str, Any], out: str) -> dict[str, Any]:
    """One completion's checks: readable, the task's rules, and for reactions the direction against the sale's."""
    task, c = ex["task"], context(ex)
    row = parse(task, out)
    res: dict[str, Any] = {"readable": row is not None}
    if row is None:
        return res
    if task == "open":
        res["offer"] = row["offer"]
        res["spoken_ok"] = stories._spoken_problem(row["says"], ex["name"]) is None
    elif task == "ask":
        res["new_question"] = row["says"].strip().lower() not in c["asked"]
        res["new_in_meaning"] = not repeats(row["says"], c["asked"])
    elif task == "react":
        res |= _react_checks(row, ex, c)
    elif task == "imagine":
        problem = stories.imagine_problem(row)
        res["rules_ok"], res["problem"] = problem is None, problem
    return res


def _react_summary(read: list[dict[str, Any]]) -> dict[str, Any]:
    """Over the readable reactions: withdrawals (model vs sale), mean offer error and the most common rule problems."""
    problems = Counter(SAID_AMOUNT.sub("$N", r["problem"]) for r in read if r["problem"])
    return {"withdrawals": {"model": sum(r["withdrew"] for r in read), "sale": sum(r["teacher_withdrew"] for r in read)},
            "mean_offer_error": round(sum(r["offer_error"] for r in read) / max(1, len(read)), 2),
            "problems": dict(problems.most_common(8))}


def summary(results: list[dict[str, Any]], examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Per task: the number of examples, the share readable, the share passing each check in RATES that applies,
    and for reactions and scenes the most common problems."""
    out: dict[str, Any] = {}
    for task in TASKS:
        rows = [r for r, e in zip(results, examples, strict=True) if e["task"] == task]
        if not rows:
            continue
        s = {"n": len(rows), "readable": round(sum(r["readable"] for r in rows) / len(rows), 3)}
        for key in RATES:
            vals = [r[key] for r in rows if key in r]
            if vals:
                s[key] = round(sum(vals) / len(vals), 3)
        if task == "react":
            s |= _react_summary([r for r in rows if r["readable"]])
        if task == "imagine":
            s["problems"] = dict(Counter(r["problem"] for r in rows if r.get("problem")).most_common(5))
        out[task] = s
    return out


def first_n(examples: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """The first n examples of each task, in file order."""
    seen: Counter[str] = Counter()
    kept = []
    for e in examples:
        seen[e["task"]] += 1
        if seen[e["task"]] <= n:
            kept.append(e)
    return kept


def samples_md(examples: list[dict[str, Any]], outputs: list[str], results: list[dict[str, Any]]) -> str:
    """A few examples per task for reading: the end of the buyer's story, the sale's line, the model's and the checks."""
    lines = []
    for task in TASKS:
        picked = [(e, o, r) for e, o, r in zip(examples, outputs, results, strict=True)
                  if e["task"] == task][:SAMPLES_PER_TASK]
        lines.append(f"## {task}\n")
        for e, o, r in picked:
            tail = e["prompt"].split("<|log|>")[-1].replace("<|end|>", "").splitlines()[-3:]
            lines += [f"**{e['name']}** (sale {e['seed']}) — " + " / ".join(tail), "",
                      f"- sale: {e['target'].replace(END, '')}", f"- model: {o}",
                      f"- checks: {json.dumps({k: v for k, v in r.items() if k != 'problem'})}"
                      + (f" — {r['problem']}" if r.get("problem") else ""), ""]
    return "\n".join(lines)


def main() -> None:
    """Generate on every validation prompt, check each line, write the report and the samples."""
    import torch

    from sellanything_ml.llm.evaluate import load_model
    from sellanything_ml.llm.pretrain import pick_device
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--data", type=Path, default=OUT_DIR / "val.jsonl")
    ap.add_argument("--n", type=int, default=0, help="at most this many examples per task (0 = all)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    device = pick_device()
    model, tok = load_model(a.checkpoint, device)
    examples = [json.loads(line) for line in a.data.read_text().splitlines()]
    if a.n:
        examples = first_n(examples, a.n)
    outputs = []
    for i, ex in enumerate(examples):
        torch.manual_seed(i)                                    # a fixed seed per prompt: the run can be repeated
        ids = torch.tensor([tok.encode(ex["prompt"])], device=device)
        with torch.no_grad():
            gen = model.generate(ids, MAX_NEW[ex["task"]], temperature=TEMPERATURE, top_k=TOP_K, stop_id=tok.eos_id)
        outputs.append(tok.decode(gen[0, ids.size(1):].tolist()).replace(END, "").strip())
    results = [check(ex, out) for ex, out in zip(examples, outputs, strict=True)]
    report = {"checkpoint": str(a.checkpoint), "examples": len(examples), "sales": len({e["seed"] for e in examples}),
              "tasks": summary(results, examples)}
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    a.out.with_name(a.out.stem + "_samples.md").write_text(samples_md(examples, outputs, results))


if __name__ == "__main__":
    main()
