"""Loading and validating items.tsv, traits.json and item_schema.json (docs/buyer-ai.md §2)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Option:
    """One possible answer to an attribute's question: its id, the seller's words, how it scales the price
    (§4 factor) and how it moves each tag (-2..2)."""
    id: str
    answer: str
    factor: float
    signals: dict[str, int]


@dataclass(frozen=True)
class Attribute:
    """Something a buyer can ask about an item (age, health, ...): the question and its 2..4 answers."""
    name: str
    question: str
    options: tuple[Option, ...]

    @property
    def tags(self) -> set[str]:
        """Every tag that any of this attribute's answers can move."""
        return {tag for option in self.options for tag in option.signals}

    def option(self, option_id: str) -> Option:
        """The answer with this id (KeyError if there is none)."""
        for option in self.options:
            if option.id == option_id:
                return option
        raise KeyError(f"{self.name} has no option {option_id!r}")


@dataclass(frozen=True)
class Item:
    """A sellable item: its Quick, Draw! class, the noun used in lines, its price range and 3..5 attributes."""
    key: str
    noun: str
    low: float
    typical: float
    high: float
    attributes: dict[str, Attribute]


@dataclass(frozen=True)
class Trait:
    """One side of a buyer ("lives in a small apartment"): how much it weighs each tag (0..2) and an optional
    budget level."""
    key: str
    persona: str
    weights: dict[str, int]
    budget: str | None
    label: str = ""          # a few words for the buyer's card ("Small apartment")
    refuses_risky: bool = False   # won't buy dangerous or black-market items (label 'risky')


@dataclass
class Traits:
    """traits.json: the tags, the traits, the budget levels (share of the typical price) and the buyer names."""
    tags: dict[str, str]
    traits: dict[str, Trait]
    budgets: dict[str, tuple[float, float]]
    default_budget_odds: dict[str, float]
    names: list[str] = field(default_factory=list)


SELLABLE = ("buy", "risky")


def load_labels(path: Path = DATA_DIR / "items.tsv") -> dict[str, str]:
    """Quick, Draw! class -> 'buy' | 'risky' (sellable, but some buyers refuse) | 'refuse'."""
    with open(path, encoding="utf-8") as f:
        return {row["en"]: row["label"] for row in csv.DictReader(f, delimiter="\t")}


def load_items(path: Path = DATA_DIR / "item_schema.json") -> dict[str, Item]:
    """item_schema.json as Item objects, by Quick, Draw! class."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = {}
    for key, spec in raw["items"].items():
        attributes = {
            name: Attribute(
                name=name,
                question=a["question"],
                options=tuple(Option(o["id"], o["answer"], float(o["factor"]), dict(o["signals"]))
                              for o in a["options"]),
            )
            for name, a in spec["attributes"].items()
        }
        p = spec["price"]
        items[key] = Item(key, spec["noun"], float(p["low"]), float(p["typical"]), float(p["high"]),
                          attributes)
    return items


def load_traits(path: Path = DATA_DIR / "traits.json") -> Traits:
    """traits.json; a trait without a label gets its key in words ("small_home" -> "Small home")."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    traits = {key: Trait(key, t["persona"], dict(t["weights"]), t.get("budget"),
                         t.get("label", key.replace("_", " ").capitalize()), t.get("refuses_risky", False))
              for key, t in raw["traits"].items()}
    budgets = {k: (float(v["min"]), float(v["max"])) for k, v in raw["budgets"].items()}
    return Traits(raw["tags"], traits, budgets, dict(raw["default_budget_odds"]), list(raw["names"]))


def _trait_errors(traits: Traits) -> list[str]:
    """Every trait weighs only known tags, 0..2, with a known budget; the budget odds and the names are sound."""
    errors: list[str] = []
    tags = set(traits.tags)
    for key, trait in traits.traits.items():
        for tag, w in trait.weights.items():
            if tag not in tags:
                errors.append(f"trait {key}: unknown tag {tag!r}")
            if not 0 <= w <= 2:
                errors.append(f"trait {key}: weight {w} for {tag} outside 0..2")
        if trait.budget is not None and trait.budget not in traits.budgets:
            errors.append(f"trait {key}: unknown budget {trait.budget!r}")
    if abs(sum(traits.default_budget_odds.values()) - 1.0) > 1e-9:
        errors.append("default_budget_odds must sum to 1")
    if len(set(traits.names)) != len(traits.names) or len(traits.names) < 5:
        errors.append("names must be unique and at least 5")
    return errors


def _option_errors(at: str, option: Option, tags: set[str]) -> list[str]:
    """One answer option: a sane price factor, a non-empty answer, known tags with signals in -2..2."""
    errors: list[str] = []
    where = f"{at}.{option.id}"
    if not 0.05 <= option.factor <= 20:
        errors.append(f"{where}: factor {option.factor} out of range")
    if not option.answer.strip():
        errors.append(f"{where}: empty answer")
    for tag, s in option.signals.items():
        if tag not in tags:
            errors.append(f"{where}: unknown tag {tag!r}")
        if s not in (-2, -1, 0, 1, 2):
            errors.append(f"{where}: signal {s} for {tag} outside -2..2")
    return errors


def _attribute_errors(at: str, attr: Attribute, tags: set[str]) -> list[str]:
    """One attribute: a real question, 2..4 distinct options, and at least one tag some buyer can care about."""
    errors: list[str] = []
    if not attr.question.endswith("?"):
        errors.append(f"{at}: question should end with '?'")
    if not 2 <= len(attr.options) <= 4:
        errors.append(f"{at}: has {len(attr.options)} options, need 2..4")
    ids = [o.id for o in attr.options]
    if len(set(ids)) != len(ids):
        errors.append(f"{at}: duplicate option ids")
    if not attr.tags:
        errors.append(f"{at}: no option moves any tag, so no buyer can care about it")
    for option in attr.options:
        errors += _option_errors(at, option, tags)
    return errors


def _item_errors(key: str, item: Item, labels: dict[str, str], tags: set[str]) -> list[str]:
    """One item: a sellable Quick, Draw! class, a sane price range and 3..5 sound attributes."""
    errors: list[str] = []
    where = f"item {key!r}"
    if key not in labels:
        errors.append(f"{where}: not a Quick, Draw! class in items.tsv")
    elif labels[key] not in SELLABLE:
        errors.append(f"{where}: labelled {labels[key]!r}, only sellable ({', '.join(SELLABLE)}) items get a schema")
    if not 0 < item.low < item.typical < item.high:
        errors.append(f"{where}: need 0 < low < typical < high")
    if not 3 <= len(item.attributes) <= 5:
        errors.append(f"{where}: has {len(item.attributes)} attributes, need 3..5")
    for name, attr in item.attributes.items():
        errors += _attribute_errors(f"{where}.{name}", attr, tags)
    return errors


def validate(items: dict[str, Item], traits: Traits, labels: dict[str, str]) -> list[str]:
    """Every rule from docs/buyer-ai.md §2 that can be checked mechanically. Empty list = valid."""
    tags = set(traits.tags)
    errors = _trait_errors(traits)
    for key, item in items.items():
        errors += _item_errors(key, item, labels, tags)
    return errors
