"""A cat sale in the game, run by the buyer GPT (B plan): the player is the seller and types every answer.

Every buyer is the same model with their own prompt, built by the same code as the training data
(buyers/sales_sft.py), so the model sees at play time exactly what it was trained on. The game only decides whose
turn it is to ask; the model writes each opening and offer, each question, each reaction with its reason and new
offer, and the three things each buyer pictures at home. Buyers have no spending limit (owner, 2026-10-12): the
engine keeps every offer the model writes, counts every line it could not read, and never rewrites or replaces a
line or samples twice to pick a better one (the project's only standard is the model itself). A format-4 model
(B1) was trained with a budget on its card, so its prompts still show one; nothing holds its offers to it.

The game speaks the same protocol as for other items (docs/protocol.md): `round_state` with a question that has no
options, typed answers (`heard` echoes what was typed), `round_sell`, and `scene`.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import stories
from .format import END
from .sales_sft import CARDS_FILE, TASK_TOKENS, Story, buyer_prompt, card, money
from .schema import DATA_DIR

ML_DIR = DATA_DIR.parent
CHECKPOINTS = (ML_DIR / "runs" / "sft_b2" / "best.pt", ML_DIR / "runs" / "sft_b1" / "best.pt")    # newest first
TYPICAL = 75                               # a typical cat's price in the sales the model learned from
MAX_NEW = {"open": 48, "ask": 48, "react": 80, "imagine": 64}
DOLLARS = re.compile(r"\$\s?(\d+)")


class BuyerGPT:
    """The fine-tuned model and its tokenizer; `say` completes one buyer's prompt."""

    def __init__(self, path: Path, device: str = "cpu", temperature: float = 0.7, top_k: int = 50):
        import torch
        from ..llm.evaluate import load_model
        self.torch = torch
        self.model, self.tok = load_model(path, torch.device(device))
        self.prompt_format = self.model.prompt_format          # 4: B1, budget on the card; 5: no budget
        self.name = "buyer_gpt_" + path.parent.name.removeprefix("sft_")     # runs/sft_b2/best.pt -> buyer_gpt_b2
        self.device, self.temperature, self.top_k = device, temperature, top_k

    def say(self, prompt: str, task: str, seed: int) -> str:
        """The model's line for this prompt, sampled with the given seed, without the prompt and <|end|>."""
        torch = self.torch
        torch.manual_seed(seed)          # same seed, same line: a whole sale can be replayed exactly
        ids = torch.tensor([self.tok.encode(prompt)], device=self.device)
        with torch.no_grad():
            out = self.model.generate(ids, MAX_NEW[task], temperature=self.temperature, top_k=self.top_k,
                                      stop_id=self.tok.eos_id)
        # keep only the new tokens (the prompt is echoed back by generate)
        return self.tok.decode(out[0, ids.size(1):].tolist()).replace(END, "").strip()


def load_default(device: str = "cpu") -> BuyerGPT | None:
    """The newest buyer GPT that has a checkpoint on disk, or None (the game then uses the phrase pools)."""
    path = next((p for p in CHECKPOINTS if p.exists()), None)
    return BuyerGPT(path, device) if path else None


def load_cards(path: Path = CARDS_FILE) -> dict[str, list[dict[str, Any]]]:
    """Persona → the buyer cards for that persona (what they care about; a budget, shown to format-4 models only).
    A card is the buyer's setup, like the persona itself; the model writes everything the buyer says and offers."""
    return json.loads(path.read_text())


def _amount(text: str) -> int | None:
    """"$55" → 55, "out" → 0 (withdrawn), anything else → None."""
    if text == "out":
        return 0
    m = DOLLARS.fullmatch(text)
    return int(m[1]) if m else None


def _parse_open(parts: list[str]) -> dict[str, Any] | None:
    """"greeting | $offer" with an offer above 0."""
    offer = _amount(parts[-1]) if len(parts) >= 2 else None
    return {"says": " | ".join(parts[:-1]), "offer": offer} if offer else None


def _parse_react(parts: list[str]) -> dict[str, Any] | None:
    """"sign | what they say | $offer" (or "out")."""
    if len(parts) < 3 or parts[0] not in stories.SIGNS:
        return None
    offer = _amount(parts[-1])
    return None if offer is None else {"sign": parts[0], "says": " | ".join(parts[1:-1]), "offer": offer}


def _parse_imagine(parts: list[str]) -> dict[str, Any] | None:
    """"thing, thing, thing | what they say"."""
    things = [t.strip() for t in parts[0].split(",") if t.strip()] if len(parts) >= 2 else []
    return {"things": things, "says": " | ".join(parts[1:])} if things else None


def parse(task: str, text: str) -> dict[str, Any] | None:
    """A model line split into its fields, or None when it does not have the trained shape."""
    if task == "ask":
        return {"says": text} if text else None
    parts = [p.strip() for p in text.split(" | ")]
    return {"open": _parse_open, "react": _parse_react, "imagine": _parse_imagine}[task](parts)


@dataclass
class CatMarket:
    """One cat sale in the game: the buyers' cards, stories and offers, the log the game shows, and the round state.
    Same interface as rounds.RoundSession, so server.py drives both the same way."""
    gpt: BuyerGPT
    rng: random.Random
    cards: dict[str, dict[str, Any]] = field(default_factory=dict)
    stories: dict[str, Story] = field(default_factory=dict)
    offers: dict[str, int] = field(default_factory=dict)
    asked: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    moods: dict[str, str] = field(default_factory=dict)
    changes: dict[str, dict[str, int]] = field(default_factory=dict)    # this round's offer change per buyer
    question: tuple[str, str] | None = None     # (asker, question) waiting for the player's answer
    sold: dict[str, Any] | None = None
    pictured: dict[str, list[dict[str, Any]]] = field(default_factory=dict)    # each buyer's scene, written once
    stats: dict[str, int] = field(default_factory=lambda: {"lines": 0, "unreadable": 0})
    item_key: str = "cat"
    label: str = "buy"

    # --- lifecycle ---------------------------------------------------------------------------

    @classmethod
    def start(cls, gpt: BuyerGPT, n_buyers: int = 4, seed: int | None = None,
              cards: dict[str, list[dict[str, Any]]] | None = None) -> CatMarket:
        """n buyers with different names and personas, each with a card from the pool; every buyer opens, then the
        first one asks. The same seed gives the same sale."""
        rng = random.Random(seed)
        # cards are the setup (who the buyer is), like a level layout; everything they say comes from the model
        pool = cards or load_cards()
        if gpt.prompt_format == 4:      # B1 learned with a budget on every card: only the cards that have one
            pool = {p: [c for c in cs if "budget" in c] for p, cs in pool.items()}
        m = cls(gpt, rng)
        personas = [p for p in stories.PERSONAS if pool.get(p)]
        for name, persona in zip(rng.sample(stories.NAMES, n_buyers), rng.sample(personas, n_buyers), strict=True):
            m.cards[name] = {"persona": persona, **rng.choice(pool[persona])}
            m.stories[name] = Story(name)
            m.moods[name] = "idle"
        for name in m.cards:
            m._open(name)
        m._next_question()
        return m

    def _say(self, name: str, task: str, offer: int | None, story: str) -> dict[str, Any] | None:
        """One line from the model for this buyer and task, parsed; None (logged and counted) if it cannot be read."""
        prompt = buyer_prompt(card(name, self.cards[name], offer, self.gpt.prompt_format), story, TASK_TOKENS[task])
        text = self.gpt.say(prompt, task, self.rng.randrange(2**31))
        self.stats["lines"] += 1
        row = parse(task, text)
        if row is None:
            # no retry and no fallback line: an unreadable line is shown as it is and counted, so a bad model looks bad
            self.stats["unreadable"] += 1
            self._log(name, text or "...", "model_unreadable")
            return None
        return row

    def _log(self, name: str, text: str, source: str) -> None:
        """A line for the game's log, with where it came from (model, model_unreadable, player)."""
        self.log.append(f"{name}: {text}")
        self.sources.append(source)

    def _open(self, name: str) -> None:
        """The buyer's greeting and first offer. One the engine cannot read leaves the buyer out of the sale (counted
        in `stats`): the engine never makes up an offer for them."""
        row = self._say(name, "open", None, "(nothing yet)")
        self.offers[name] = row["offer"] if row else 0
        if row is None or row["offer"] == 0:
            self.moods[name] = "left"
            return
        self._log(name, row["says"], "model")
        self.stories[name].opening = f"Your opening ({money(row['offer'])}): {row['says']}"

    def present(self) -> list[str]:
        """Buyers still bidding (offer above 0), in seating order."""
        return [n for n in self.cards if self.offers[n] > 0]

    def _next_question(self) -> None:
        """The next buyer who has not asked yet writes their question; none left: the questions are over."""
        self.question = None
        # the only thing the engine decides: whose turn it is (the next buyer still here who hasn't asked yet)
        for name in self.present():
            if name in self.asked:
                continue
            self.asked.append(name)
            row = self._say(name, "ask", self.offers[name], self.stories[name].text())
            if row:
                self._log(name, row["says"], "model")
                self.question = (name, row["says"])
                self.moods[name] = "talk"
                return

    # --- the player ----------------------------------------------------------------------------

    def answer(self, option: str) -> list[str]:
        """Picking an option does not apply here: these buyers ask in their own words, so the player types."""
        raise ValueError("this buyer asks in their own words; type your answer")

    def answer_text(self, text: str) -> tuple[list[str], dict[str, Any]]:
        """The player's typed answer: every buyer still here reacts, then the next buyer asks."""
        if self.question is None:
            raise ValueError("no question is waiting")
        text = " ".join(text.split())[:120]       # same limit as the game's text box
        if not text:
            raise ValueError("empty answer")
        start = len(self.log)
        asker, q = self.question
        self.log.append(f"You: {text}")
        self.sources.append("player")
        # the player's words go into every buyer's story exactly like the seller lines in the training sales
        for s in self.stories.values():
            s.asked(asker, q)
            s.rounds[-1].append(f"Seller: {text}")
        self.changes = {}
        for name in self.present():
            self._react(name)
        self._next_question()
        return self.log[start:], {"option": None, "answer": text, "rule": "model"}

    def _react(self, name: str) -> None:
        """One buyer reacts to the answer just typed; their offer and mood follow what the model wrote."""
        old = self.offers[name]
        row = self._say(name, "react", old, self.stories[name].text(live=True))
        if row is None:
            self.moods[name] = "idle"
            return
        self._log(name, row["says"], "model")
        self.stories[name].rounds[-1].append(f"You ({money(old)} -> {money(row['offer'])}): {row['says']}")
        self.offers[name] = row["offer"]
        self.changes[name] = {"amount": row["offer"] - old}
        self.moods[name] = ("left" if row["offer"] == 0 else "happy" if row["offer"] > old
                            else "unhappy" if row["offer"] < old else "idle")

    def sell(self, buyer: str) -> dict[str, Any]:
        """Sell to a buyer still bidding, once the questions are over, at their current offer."""
        if self.question is not None:
            raise ValueError("the round isn't finished")
        if buyer not in self.present():
            raise ValueError(f"{buyer!r} has no offer")
        self.sold = {"buyer": buyer, "price": self.offers[buyer], "typical": TYPICAL}
        self.moods[buyer] = "won"
        return self.sold

    def scene(self, buyer: str) -> list[dict[str, Any]]:
        """The cat and the three things this buyer pictures with it (written by the model), in the scene layout.
        Where a thing goes is display: its zone in props.json, or beside the cat."""
        from . import scenes
        if buyer not in self.present():
            raise ValueError(f"no buyer named {buyer} is still here")
        if buyer in self.pictured:
            return self.pictured[buyer]
        row = self._say(buyer, "imagine", self.offers[buyer], self.stories[buyer].text())
        props, _ = scenes.tables()
        objects = [{"label": "cat", "role": "item", "zone": "item", "box": list(scenes.ITEM_BOX), "share": 1.0,
                    "reason": ""}]
        used: set[str] = set()
        for thing in (row["things"] if row else [])[:scenes.MAX_PROPS]:
            prop = props.get(thing)
            zone = prop.zone if prop and prop.zone not in used else next(
                z for z in scenes.ZONES if z not in used)
            used.add(zone)
            objects.append({"label": thing, "role": "prop", "zone": zone, "box": list(scenes.ZONES[zone]),
                            "share": scenes.SIZES[prop.size if prop else "medium"], "reason": row["says"]})
        self.pictured[buyer] = objects
        return objects

    # --- what the game draws -------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """The round_state the game draws (docs/protocol.md); `voice` names the model that wrote the lines."""
        q = None if self.question is None else {"asker": self.question[0], "attribute": "", "options": []}
        return {
            "item": "cat", "label": "buy", "noun": "cat", "finished": self.question is None,
            "buyers": [{"name": n, "persona": f"{n}, who {c['persona']}.",
                        "offer": self.offers[n] if self.offers[n] > 0 else None, "present": self.offers[n] > 0,
                        "mood": self.moods[n], "traits": c["cares_about"], "change": self.changes.get(n)}
                       for n, c in self.cards.items()],
            "log": self.log, "sources": self.sources, "question": q, "sold": self.sold, "voice": self.gpt.name,
            "model_stats": dict(self.stats)}
