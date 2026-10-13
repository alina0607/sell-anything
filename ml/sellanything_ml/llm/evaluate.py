"""Buyer-model evaluation (docs/buyer-ai.md §9).

    python -m sellanything_ml.llm.evaluate --checkpoint runs/sft/best.pt --split test_heldout

Greedy-decodes each example's prompt and scores the output against the engine's label.

`parse` (which option a typed answer means) is scored on every parse example of a split, next to
the rule matcher (buyers/answers.py) on the same answers, with three outcomes kept apart as in
buyer-ai.md §10: right, can't tell (asks again), wrong option (worse: the game believes it).

    python -m sellanything_ml.llm.evaluate --checkpoint runs/sft_v6/best.pt --parse
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from ..buyers import format as fmt
from ..buyers.answers import match
from ..buyers.datagen import load_split
from ..buyers.schema import Attribute, Item, load_items
from .model import GPT, GPTConfig
from .tokenizer import TOKENIZER_DIR, Tokenizer

Example = dict[str, Any]                 # one row of a buyers_* split
Report = dict[str, dict[str, float]]     # task or group -> {"n": count, measure: share}
LINE_TOKENS = 64        # longest buyer line we decode
PARSE_TOKENS = 4        # a parse answer is one letter (or "none") and <|end|>
SHOWN_EXAMPLES = 12     # prompts/outputs kept in the run report for reading


def load_model(path: Path, device: torch.device) -> tuple[GPT, Tokenizer]:
    """The model is tagged with `prompt_format`: the format its training data used (1 if unrecorded)."""
    state = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**state["config"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    model.prompt_format = state.get("prompt_format", 1)
    model.tasks = state.get("tasks", ["ask", "react", "refuse"])
    model.parse_ready = state.get("parse_ready", False)      # set by llm/export.py from the evaluation
    return model, Tokenizer(TOKENIZER_DIR / f"{state['tokenizer']}.model")


@torch.no_grad()
def complete(model: GPT, tok: Tokenizer, prompt: str, max_new: int = LINE_TOKENS) -> str:
    """Greedy continuation of the prompt, up to <|end|> or max_new tokens, without the prompt."""
    ids = torch.tensor([tok.encode(prompt)], device=next(model.parameters()).device)
    out = model.generate(ids, max_new, temperature=0, stop_id=tok.eos_id)
    return tok.decode(out[0, ids.size(1):].tolist())


def _read_option(model: GPT, tok: Tokenizer, prompt: str, ids: list[str]) -> str | None | bool:
    """The option id the model picks for a parse prompt whose options are listed in `ids` order; None for
    "none", False if the output is malformed."""
    got = fmt.read_parse(complete(model, tok, prompt, max_new=PARSE_TOKENS), len(ids))
    return got if got is None or got is False else ids[got]   # `in (None, False)` would take 0 for False


def _score_parse(c: Counter[str], ex: Example, out: str) -> None:
    """Counts one parse answer: valid, the right option, or a wrong option (worse than "none")."""
    got = fmt.read_parse(out, len(ex["order"]))
    if got is False:
        return
    c["valid"] += 1
    got = None if got is None else ex["order"][got]
    c["exact"] += got == ex["option"]
    c["wrong_option"] += got is not None and got != ex["option"]


def _score_react(c: Counter[str], ex: Example, out: str) -> int | None:
    """Counts one reaction; returns |Δ error| when the output parses."""
    parsed = fmt.parse_react(out)
    if parsed is None:
        return None
    c["valid"] += 1
    delta = parsed[2]
    c["delta_exact"] += delta == ex["delta"]
    c["delta_sign"] += (delta > 0) - (delta < 0) == (ex["delta"] > 0) - (ex["delta"] < 0)
    c["speaks_match"] += (parsed[1] is not None) == ex["speaks"]
    return abs(delta - ex["delta"])


def _score_line(c: Counter[str], ex: Example, out: str, items: dict[str, Item]) -> None:
    """An ask or refuse line: right buyer name, and for a question, still on the asked topic."""
    parsed = fmt.parse_line(out)
    if parsed is None:
        return
    c["valid"] += 1
    c["name_match"] += parsed[0] == ex["buyer"]
    if ex["task"] == "ask" and ex["item"] in items:
        item = items[ex["item"]]
        question = item.attributes[ex["attr"]].question
        c["on_topic"] += fmt.on_topic(fmt.fill_slot(parsed[1], item.noun), question, item.noun)


def score(examples: list[Example], outputs: list[str], items: dict[str, Item]) -> Report:
    """Per task, the share of outputs that parse ("valid") and that agree with the engine's label: the option
    for parse, Δ (exact, sign, mean error) and speaking for react, the name and topic for ask and refuse."""
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    abs_err = []
    for ex, out in zip(examples, outputs):
        c = by_task[ex["task"]]
        c["n"] += 1
        if ex["task"] == "parse":
            _score_parse(c, ex, out)
        elif ex["task"] == "react":
            if (err := _score_react(c, ex, out)) is not None:
                abs_err.append(err)
        else:
            _score_line(c, ex, out, items)
    report: Report = {}
    for task, c in by_task.items():
        n = c.pop("n")
        report[task] = {"n": n} | {k: round(v / n, 4) for k, v in c.items()}
    if abs_err:
        report["react"]["delta_mae"] = round(sum(abs_err) / len(abs_err), 4)
    return report


def run(model: GPT, tok: Tokenizer, split: str, n: int, data_dir: Path | None = None) -> dict[str, Any]:
    """Score a stratified sample of n examples of a split (always the same sample: seed 0)."""
    rows = load_split(split, data_dir) if data_dir else load_split(split)
    rng = random.Random(0)
    # Stratify so every task is represented even in small samples.
    by_task: dict[str, list[Example]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    sample = []
    for task, group in by_task.items():
        k = max(1, round(n * len(group) / len(rows)))
        sample += rng.sample(group, min(k, len(group)))
    outputs = [complete(model, tok, ex["prompt"], max_new=LINE_TOKENS) for ex in sample]
    return {"split": split, "report": score(sample, outputs, load_items()),
            "examples": [{"prompt": ex["prompt"][-300:], "target": ex["target"], "output": out}
                         for ex, out in list(zip(sample, outputs))[:SHOWN_EXAMPLES]]}


def _outcome(picked: str | None | bool, target: str | None) -> str:
    """right, can't tell (None: the buyer asks again), wrong option (the game would believe it) or malformed."""
    if picked is False:
        return "malformed"
    if picked == target:
        return "right"
    return "can't tell" if picked is None else "wrong option"


def parse_choice(model: GPT, tok: Tokenizer, noun: str, attr: Attribute, said: str) -> str | None | bool:
    """The option id the model reads in `said` (options in schema order), None, or False if malformed."""
    prompt = fmt.parse_prompt(noun, attr.question, [o.answer for o in attr.options], said)
    return _read_option(model, tok, prompt, [o.id for o in attr.options])


@torch.no_grad()
def parse_report(model: GPT | None, tok: Tokenizer | None, examples: list[Example]) -> Report:
    """Model (as trained: each example's own option order) and rule matcher on the same answers, overall and
    split by typo/clean and by whether the answer means an option at all."""
    items = load_items()
    res: dict[str, Counter[str]] = defaultdict(Counter)
    for ex in examples:
        attr = items[ex["item"]].attributes[ex["attr"]]
        groups = ["all", "typo" if ex.get("typo") else "clean", "none" if ex["option"] is None else "option"]
        if model is not None:
            got = _read_option(model, tok, ex["prompt"], ex["order"])
            for g in groups:
                res[f"model {g}"][_outcome(got, ex["option"])] += 1
        rule = match(ex["said"], attr).option
        for g in groups:
            res[f"rules {g}"][_outcome(rule, ex["option"])] += 1
    return {k: {"n": sum(c.values())} | {o: round(v / sum(c.values()), 4) for o, v in sorted(c.items())}
            for k, c in sorted(res.items())}


def _fixture_lines(path: Path) -> list[list[str]]:
    """The rows of a typed-answer fixture (item, attribute, typed, expected), without blanks and comments."""
    return [line.split("\t") for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def fixture_report(model: GPT | None, tok: Tokenizer | None, path: Path) -> dict[str, Any]:
    """Hand-written typed answers (tests/fixtures/typed_answers_*.tsv): item, attribute, typed, expected
    ("-" when no option fits). Every row with the model's and the rules' pick, and the outcome counts."""
    items = load_items()
    res: dict[str, Counter[str]] = defaultdict(Counter)
    rows = []
    for key, attr_name, typed, expected in _fixture_lines(path):
        item = items[key]
        attr = item.attributes[attr_name]
        target = None if expected == "-" else expected
        row = {"item": key, "attr": attr_name, "typed": typed, "expected": expected}
        if model is not None:
            got = parse_choice(model, tok, item.noun, attr, typed)
            res["model"][_outcome(got, target)] += 1
            row["model"] = got
        rule = match(typed, attr).option
        res["rules"][_outcome(rule, target)] += 1
        row["rules"] = rule
        rows.append(row)
    return {"file": path.name, "rows": rows,
            "summary": {k: {"n": sum(c.values())} | dict(sorted(c.items())) for k, c in res.items()}}


def main() -> None:
    """Score a split (default) or, with --parse, the parse task on a split and on the fixtures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test_heldout")
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--data", type=Path, default=None, help="dataset dir (default: buyers_v5)")
    parser.add_argument("--parse", action="store_true", help="score parse on every parse example and the fixtures")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    model, tok = load_model(args.checkpoint, device)
    if not args.parse:
        print(json.dumps(run(model, tok, args.split, args.n, data_dir=args.data), indent=2))
        return
    rows = [r for r in load_split(args.split, args.data) if r["task"] == "parse"]
    fixtures = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    print(json.dumps({"split": args.split, "parse": parse_report(model, tok, rows),
                      "fixtures": {p.name: fixture_report(model, tok, p)["summary"]
                                   for p in sorted(fixtures.glob("typed_answers_*.tsv"))}}, indent=2))


if __name__ == "__main__":
    main()
