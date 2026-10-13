"""Buyer GPT, B plan: the official cat sales as supervised fine-tuning examples, one per decision a buyer makes
(devlog 2026-10-11 §3; owner decisions in HANDOFF.md).

    python -m sellanything_ml.buyers.sales_sft            # → data/processed/buyers_b2/{train,val}.jsonl, meta.json

Every example is what ONE buyer knows at that moment and what that buyer did next:

    open     their card (no offer yet)                           → greeting | $offer
    ask      card + their own story so far                       → the question they ask
    react    card + story + this round's question and answer     → sign | what they say | $offer (or "out")
    imagine  card + story after the last round                   → three drawable things | what they say

The story is the buyer's own opening, the earlier questions (only the questions, so they do not ask one again;
owner's idea, 2026-10-11) and the last K rounds in full: who asked what, the seller's answer and their own
reaction. Buyers never see other buyers' reactions (owner, 2026-10-11). The card carries the offer the buyer holds,
so facts from older rounds reach the prompt through it.

Nothing in a prompt gives the answer away: the engine only decides whose turn it is to ask, and the model writes
the question, the reaction with its reason, the new offer and the scene (memory: buyer-model-decides).

Sales are split by seed: about one in ten (by a hash of the seed, so adding sales never moves one) is validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .format import ASK, END, IMAGINE, ITEM, LOG, OPEN, PERSONA, REACT
from .schema import DATA_DIR

PROMPT_FORMAT = 5          # 4 (buyer GPT B1): the card shows a budget; 5: no budget (owner, 2026-10-12: buyers have
                           # no spending limit, their raises and cuts only have to be reasonable)
K = 2                                      # full rounds in the story (fits 768 tokens: devlog 2026-10-11 §3)
WRITTEN_SALES = DATA_DIR.parents[1] / "data" / "written_sales"     # sales written to the prompts, not by Qwen
SOURCES_B1 = (DATA_DIR / "processed" / "buyer_sales_v9_official.jsonl",    # Qwen v9, every line reviewed and corrected
              WRITTEN_SALES / "sales.jsonl")                              # seeds 2048-2099
# B2 adds the sales written to the format-5 prompts with the quality gates (seeds 2200-, no budget, a note per answer)
SOURCES = SOURCES_B1 + (WRITTEN_SALES / "sales_b2.jsonl",)
OUT_DIR = DATA_DIR / "processed" / "buyers_b2"
CARDS_FILE = DATA_DIR / "cat_buyer_cards.json"   # persona → the cards written for it (the game's buyer setup)
VAL_EVERY = 10                             # one sale in VAL_EVERY (by seed hash) is validation
Example = dict[str, Any]


def is_val(seed: int) -> bool:
    """Whether the sale on this seed is in the validation split (about one in VAL_EVERY)."""
    # hash the seed instead of shuffling: a sale lands on the same side every time, even after more sales are added,
    # so the validation set I compare versions on never quietly changes
    return int(hashlib.sha1(str(seed).encode()).hexdigest(), 16) % VAL_EVERY == 0


def money(offer: int) -> str:
    """An offer as the model reads and writes it: "$70", or "out" for 0."""
    # an offer of 0 means the buyer walked away, and "out" reads better than "$0" for the model and for me
    return f"${offer}" if offer else "out"


def card(name: str, c: dict[str, Any], offer: int | None, prompt_format: int = PROMPT_FORMAT) -> str:
    """Who the buyer is; with the offer they hold, except before their opening. Format 4 also shows the budget."""
    budget = f" | budget ${c['budget']}" if prompt_format == 4 else ""
    now = f" | offer now ${offer}" if offer is not None else ""
    return f"{name}, who {c['persona']} | cares about: {', '.join(c['cares_about'])}{budget}{now}"


class Story:
    """One buyer's own view of a sale as it goes: their opening, every round's question and answer, and their own
    reactions."""

    def __init__(self, name: str):
        self.name = name
        self.opening = ""
        self.rounds: list[list[str]] = []          # per round: question line, answer line, own reaction line
        self.questions: list[str] = []

    def asked(self, asker: str, question: str) -> None:
        """A new round starts with this question ("You asked" when it is this buyer's own)."""
        who = "You" if asker == self.name else asker
        self.rounds.append([f"{who} asked: {question}"])
        self.questions.append(question)

    def text(self, live: bool = False, k: int = K) -> str:
        """The story so far: opening, the questions of older rounds, the last k finished rounds and, if `live`,
        the round under way (its question and answer, before this buyer reacts)."""
        done = self.rounds[:-1] if live else self.rounds
        lines = [self.opening] if self.opening else []
        # rounds older than k only keep their questions: the buyer still knows what was asked (so it won't ask it
        # again), but the answers are gone. This is the "forgets after two rounds" problem B1 has (devlog 10-12 §6)
        older = self.questions[:max(0, len(done) - k)]
        if older:
            lines.append("Asked earlier: " + " / ".join(older))
        for r in done[len(done) - min(k, len(done)):] + (self.rounds[-1:] if live else []):
            lines += r
        return "\n".join(lines) or "(nothing yet)"


def buyer_prompt(card_line: str, story: str, task_token: str) -> str:
    """What one buyer sees: the item, their card, their story so far, and the token of the task to do next."""
    return f"{ITEM}cat{END}{PERSONA}{card_line}{END}{LOG}{story}{END}{task_token}"


class _Walk:
    """Goes through one sale's events, keeping each buyer's story and offer, and collects the examples."""

    def __init__(self, sale: dict[str, Any], prompt_format: int = PROMPT_FORMAT):
        self.cards, self.seed, self.prompt_format = sale["cards"], sale["seed"], prompt_format
        self.stories = {n: Story(n) for n in self.cards}
        self.offers: dict[str, int] = {}
        self.out: list[Example] = []
        self.current: tuple[str, str] | None = None          # (asker, question) of the round under way
        self.meta = {"seed": self.seed, "author": sale.get("author", "qwen_v9_edited")}

    def add(self, task: str, name: str, offer: int | None, story: str, target: str) -> None:
        """One example: this buyer's prompt for the task and what they did (ending <|end|>)."""
        prompt = buyer_prompt(card(name, self.cards[name], offer, self.prompt_format), story, TASK_TOKENS[task])
        self.out.append({"task": task, "name": name, "prompt": prompt, "target": target + END, **self.meta})

    def open(self, e: dict[str, Any]) -> None:
        """The greeting and first offer, written from the card alone."""
        name = e["name"]
        self.add("open", name, None, "(nothing yet)", f"{e['says']} | {money(e['offer'])}")
        self.offers[name] = e["offer"]
        self.stories[name].opening = f"Your opening ({money(e['offer'])}): {e['says']}"

    def ask(self, e: dict[str, Any]) -> None:
        """The question this buyer asks, from their story so far; it starts the round under way."""
        # the offer comes from the reactions: v9's ask events kept the offer from before the hand corrections
        # (293 of 451 differ; the reaction chain itself has no break)
        name = e["name"]
        self.add("ask", name, self.offers[name], self.stories[name].text(), e["says"])
        self.current = (name, e["says"])

    def answer(self, e: dict[str, Any]) -> None:
        """The seller's answer goes into every buyer's story with the question (no example: the player is the
        seller)."""
        # everyone hears the question and the answer, but each buyer only ever sees their own reactions
        if self.current is None or self.current[0] != e["to"]:
            raise ValueError(f"sale {self.seed}: an answer to {e['to']} with no question from them")
        for s in self.stories.values():
            s.asked(*self.current)
            s.rounds[-1].append(f"Seller: {e['says']}")

    def react(self, e: dict[str, Any]) -> None:
        """One buyer's reaction to the answer: sign, what they say and the new offer."""
        name, s = e["name"], self.stories[e["name"]]
        # sanity check: if the offer chain does not add up, the sale file is broken and I'd rather crash than train
        # on it
        if e["old"] != self.offers[name]:
            raise ValueError(f"sale {self.seed}: {name} reacts from ${e['old']}, the record says ${self.offers[name]}")
        # the target puts the sign first, then the spoken reason, then the number, so the reason is written before
        # the model commits to an amount
        self.add("react", name, e["old"], s.text(live=True), f"{e['sign']} | {e['says']} | {money(e['offer'])}")
        s.rounds[-1].append(f"You ({money(e['old'])} -> {money(e['offer'])}): {e['says']}")
        self.offers[name] = e["offer"]

    def imagine(self, e: dict[str, Any]) -> None:
        """The three things this buyer pictures with the cat at home, and what they say about them."""
        name = e["name"]
        self.add("imagine", name, self.offers[name], self.stories[name].text(), f"{', '.join(e['things'])} | {e['says']}")

    def sold(self, e: dict[str, Any]) -> None:
        """The sale's end: the player picks the buyer, so there is nothing for a buyer to learn."""


TASK_TOKENS = {"open": OPEN, "ask": ASK, "react": REACT, "imagine": IMAGINE}
STEPS = ("open", "ask", "answer", "react", "imagine", "sold")


def examples(sale: dict[str, Any], prompt_format: int = PROMPT_FORMAT) -> list[Example]:
    """Every decision of every buyer in one sale, in order, checked against the sale's own bookkeeping."""
    walk = _Walk(sale, prompt_format)
    for e in sale["events"]:
        if e["kind"] not in STEPS:      # getattr below would happily call the wrong thing, so check the name first
            raise ValueError(f"sale {walk.seed}: unknown event kind {e['kind']!r}")
        getattr(walk, e["kind"])(e)
    return walk.out


def write_cards(sales: list[dict[str, Any]], path: Path = CARDS_FILE) -> None:
    """Every buyer card in the sales, by persona, for setting up buyers in the game (buyers/cat_market.py)."""
    cards: dict[str, list[dict[str, Any]]] = {}
    for sale in sales:
        for c in sale["cards"].values():
            entry = {"cares_about": c["cares_about"]}
            if "budget" in c:
                entry["budget"] = c["budget"]                                   # format-4 models only
            if entry not in cards.setdefault(c["persona"], []):
                cards[c["persona"]].append(entry)
    path.write_text(json.dumps(dict(sorted(cards.items())), indent=1))


def build(sources: tuple[Path, ...] = SOURCES, out_dir: Path = OUT_DIR, tokenizer: Path | None = None,
          prompt_format: int = PROMPT_FORMAT) -> dict[str, Any]:
    """Write train.jsonl and val.jsonl (and meta.json) from the sales; report counts and, with a tokenizer, the
    prompt + target lengths in tokens."""
    sales = [json.loads(line) for path in sources for line in path.read_text().splitlines() if line.strip()]
    seeds = [s["seed"] for s in sales]
    if len(seeds) != len(set(seeds)):          # two sales on one seed = the same cat and buyers twice
        raise ValueError("a seed appears in two sales")
    splits: dict[str, list[Example]] = {"train": [], "val": []}
    for sale in sales:
        splits["val" if is_val(sale["seed"]) else "train"].extend(examples(sale, prompt_format))
    out_dir.mkdir(parents=True, exist_ok=True)
    if prompt_format == PROMPT_FORMAT:          # the game's card pool follows the current format's sales only
        write_cards(sales)
    for split, rows in splits.items():
        (out_dir / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report: dict[str, Any] = {
        "sales": {split: len({r["seed"] for r in rows}) for split, rows in splits.items()},
        "examples": {split: dict(Counter(r["task"] for r in rows)) for split, rows in splits.items()},
        "authors": dict(Counter(s.get("author", "qwen_v9_edited") for s in sales))}
    if tokenizer:
        from ..llm.tokenizer import Tokenizer
        tok = Tokenizer(tokenizer)
        lengths: dict[str, list[int]] = {}
        for r in splits["train"] + splits["val"]:
            lengths.setdefault(r["task"], []).append(len(tok.encode(r["prompt"] + r["target"])))
        report["tokens"] = {task: {"median": sorted(v)[len(v) // 2], "p95": sorted(v)[int(len(v) * 0.95)],
                                   "max": max(v), "over_512": sum(x > 512 for x in v),
                                   "over_768": sum(x > 768 for x in v)} for task, v in lengths.items()}
    meta = {"prompt_format": prompt_format, "tasks": ["open", "ask", "react", "imagine"], "k": K,
            "sources": [str(p) for p in sources], **report}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return report


def main() -> None:
    """Build the current format's data (or, with --prompt-format 4, B1's) and print the report."""
    from ..llm.tokenizer import TOKENIZER_DIR
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR / "sp8192.model")
    ap.add_argument("--prompt-format", type=int, default=PROMPT_FORMAT, choices=[4, 5],
                    help="4 rebuilds buyer GPT B1's data (budget on the card)")
    a = ap.parse_args()
    sources = SOURCES_B1 if a.prompt_format == 4 else SOURCES      # B1's data is exactly its two sources
    print(json.dumps(build(sources, a.out, a.tokenizer, a.prompt_format), indent=1))


if __name__ == "__main__":
    main()
