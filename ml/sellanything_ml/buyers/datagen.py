"""Buyer-model training data v1: engine skeleton + phrase pools (docs/buyer-ai.md §8).

    python -m sellanything_ml.buyers.datagen --scenes-per-item 60

Writes <out>/{train,val,test_seen,test_heldout}.jsonl (one example per line), scenes.jsonl (full
rounds, for debugging) and stats.json. Every label comes from the engine; the text is decoration
that is chosen to agree with the label. ml/data/item_splits.json fixes which items are held out.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import format as fmt
from . import phrases
from .engine import Buyer, Reaction, Round, make_buyers, refuses_risky
from .scenes import load_groups
from .schema import DATA_DIR, Attribute, Item, Option, Traits, load_items, load_labels, load_traits

SPLITS_FILE = DATA_DIR / "item_splits.json"
# The pipeline: datagen writes RAW_DIR, buyers/augment.py renames items in part of it into BUYERS_DIR,
# which training, evaluation and the baselines read.
RAW_DIR = DATA_DIR / "processed" / "buyers_v5_raw"
BUYERS_DIR = DATA_DIR / "processed" / "buyers_v5"
HELDOUT_FRACTION = 0.15
SHOW_ATTRS_P = 0.8          # fraction of scenes whose prompts list the attributes
KEEP_SILENT_ZERO_P = 0.3    # downsample silent Δ = 0 reactions, otherwise they dominate the labels
REFUSALS_PER_ITEM = 20
SPLITS = ("train", "val", "test_seen", "test_heldout")
SPLIT_EVERY, VAL_SLOT, TEST_SLOT = 10, 8, 9   # of every ten scenes of a training item: #8 to val, #9 to test_seen

Example = dict[str, Any]      # one training example (or scene) as written to the .jsonl files


def is_current(data_dir: Path) -> bool:
    """Would today's code write this dataset? Re-simulates the first item with the recorded seed and
    compares its scenes and training examples with what is on disk, so a change that alters the data
    (engine rules, phrases, prompt format) is caught and one that doesn't (a comment) is not.
    Examples renamed by augment.py are skipped; the rest of an augmented set must still match.
    Datasets without a seed in meta.json are compared as seed 0, 60 scenes per item, format 1."""
    meta = json.loads((data_dir / "meta.json").read_text()) if (data_dir / "meta.json").exists() else {}
    rng = random.Random(meta.get("seed", 0))
    per_item, prompt_format = meta.get("scenes_per_item", 60), meta.get("prompt_format", 1)
    labels = load_labels()
    first = item_splits(labels)["buy_train"][0]
    items, traits = load_items(), load_traits()
    scenes, train = [], []
    for i in range(per_item):
        scene, examples = simulate_scene(items[first], traits, rng, prompt_format, risky=labels[first] == "risky")
        scenes.append(scene)
        if split_of(i, heldout=False) == "train":
            train += examples
    with open(data_dir / "scenes.jsonl") as f:
        stored_scenes = [json.loads(next(f)) for _ in scenes]
    with open(data_dir / "train.jsonl") as f:
        stored_train = [json.loads(next(f)) for _ in train]
    for s in stored_scenes:
        s.pop("split", None)
    # Rows renamed by buyers/augment.py differ on purpose; every other row must match exactly.
    kept = [(stored, fresh) for stored, fresh in zip(stored_train, train) if "swapped_to" not in stored]
    return stored_scenes == scenes and all(stored == fresh for stored, fresh in kept)


def prompt_format_of(data_dir: Path) -> int:
    """The prompt format a dataset was written in (datasets before meta.json are format 1)."""
    meta = data_dir / "meta.json"
    return json.loads(meta.read_text())["prompt_format"] if meta.exists() else 1


def load_split(split: str, data_dir: Path = BUYERS_DIR) -> list[Example]:
    """The examples of one split ("train", "val", "test_seen", "test_heldout")."""
    with open(data_dir / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def item_splits(labels: dict[str, str], seed: int = 0) -> dict[str, list[str]]:
    """Fixed train / held-out split of buy and refuse items, created once and then reused."""
    if SPLITS_FILE.exists():
        return json.loads(SPLITS_FILE.read_text())
    rng = random.Random(seed)
    out: dict[str, list[str]] = {}
    for label in ("buy", "refuse"):
        keys = sorted(k for k, v in labels.items() if v == label)
        rng.shuffle(keys)
        n_held = round(len(keys) * HELDOUT_FRACTION)
        out[f"{label}_heldout"] = sorted(keys[:n_held])
        out[f"{label}_train"] = sorted(keys[n_held:])
    SPLITS_FILE.write_text(json.dumps(out, indent=1) + "\n")
    return out


@lru_cache(maxsize=1)
def _groups() -> dict[str, str]:
    """Item key -> group (animal, food, ...), read once."""
    return load_groups()


def reason_topic(item: Item, attr: Attribute) -> str | None:
    """phrases.topic for this item's attribute: health, freshness or an animal's age, or None."""
    return phrases.topic(attr.name, _groups().get(item.key, ""), item.attributes)


def reason_tag(weights: dict[str, int], signals: dict[str, int]) -> str | None:
    """The tag that contributed most to Δ (largest |weight × signal|), or None if Δ had no cause."""
    scored = [(abs(weights.get(t, 0) * s), t) for t, s in signals.items() if weights.get(t, 0) * s]
    return max(scored)[1] if scored else None


def ask_reason_tag(buyer: Buyer, attr: Attribute) -> str | None:
    """The tag this buyer cares about most among the ones the attribute moves, or None if it moves none of them."""
    scored = [(buyer.weights.get(t, 0), t) for t in attr.tags if buyer.weights.get(t, 0) > 0]
    return max(scored)[1] if scored else None


class _Scene:
    """One simulated round of the engine, written down as a log, training examples and events.

    The random generator is consumed in a fixed order (buyers, shown attributes, then per question the ask phrase,
    the answer, the reaction phrases and the bid phrases), so a seed always gives the same scene."""

    def __init__(self, item: Item, traits: Traits, rng: random.Random, prompt_format: int) -> None:
        self.item, self.rng, self.prompt_format = item, rng, prompt_format
        self.buyers = make_buyers(item, traits, rng.randint(3, 5), rng)
        self.rnd = Round(item, self.buyers)
        self.personas = {b.name: b.persona(traits) for b in self.buyers}
        self.attrs = list(item.attributes) if rng.random() < SHOW_ATTRS_P else None
        self.by_name = {b.name: b for b in self.buyers}
        self.names = fmt.item_names(item.key, item.noun)
        self.log: list[str] = []
        self.examples: list[Example] = []
        self.events: list[Example] = []

    def said(self, text: str) -> str:
        """What the model learns to write: from format 3 on, the item's name is ITEM_SLOT."""
        return fmt.to_slot(text, self.names) if self.prompt_format >= 3 else text

    def base(self, task: str, buyer: Buyer) -> Example:
        """The fields every example of this scene shares."""
        return {"task": task, "item": self.item.key, "buyer": buyer.name, "traits": buyer.traits,
                "show_attrs": self.attrs is not None}

    def view(self, buyer: Buyer, log: list[str]) -> fmt.BuyerView:
        """What this buyer sees in a prompt: the item, the attributes (if shown), their persona and the log."""
        return fmt.BuyerView(self.item.noun, self.attrs, self.personas[buyer.name], log)

    def refuse_risky(self, traits: Traits) -> None:
        """As in the game (rounds.py): the wary say why and leave before the questions."""
        for b in self.buyers:
            if (trait := refuses_risky(b, traits)) is not None:
                self.rnd.dismiss(b)
                self.log.append(f"{b.name}: {phrases.risky_refusal(trait, self.rng)}")

    def ask(self, asker: Buyer, attr: Attribute) -> None:
        """The asker's question, with a reason when they care about a tag it moves."""
        line = phrases.ask_with_reason(attr.question, ask_reason_tag(asker, attr), self.rng,
                                       reason_topic(self.item, attr))
        question = self.said(attr.question) if self.prompt_format >= 2 else None
        self.examples.append(self.base("ask", asker) | {
            "attr": attr.name,
            "prompt": fmt.ask_prompt(self.view(asker, self.log), attr.name, question),
            "target": fmt.ask_target(asker.name, self.said(line))})
        self.log.append(f"{asker.name}: {line}")

    def reaction_line(self, r: Reaction, option: Option, attr: Attribute) -> str | None:
        """What a buyer says after the answer: why they leave, why the offer moved, or nothing."""
        if r.left:
            return phrases.leave(self.item.noun, self.rng)
        if r.speaks:
            tag = reason_tag(self.by_name[r.buyer].weights, option.signals)
            return phrases.react_with_reason(r.delta, self.item.noun, option.factor, tag, self.rng,
                                             reason_topic(self.item, attr))
        return None

    def answer(self, asker: Buyer, attr: Attribute) -> None:
        """A random true answer, every buyer's reaction to it, and the bids that follow."""
        option = self.rng.choice(attr.options)
        self.log.append(f"You: {phrases.answer(option.answer, self.rng)}")
        snapshot = list(self.log)
        reactions, bids = self.rnd.answer(asker, attr, option.id)
        spoken = []
        for r in reactions:
            text = self.reaction_line(r, option, attr)
            if text is None and r.delta == 0 and self.rng.random() > KEEP_SILENT_ZERO_P:
                continue
            b = self.by_name[r.buyer]
            self.examples.append(self.base("react", b) | {
                "attr": attr.name, "option": option.id, "delta": r.delta, "speaks": text is not None,
                "prompt": fmt.react_prompt(self.view(b, snapshot)),
                "target": fmt.react_target(b.name, self.said(text) if text else text, r.delta)})
            if text:
                spoken.append(f"{b.name}: {text}")
        self.log.extend(spoken)
        for bid in bids:
            self.log.append(f"{bid.buyer}: {phrases.raise_bid(round(bid.amount, 2), self.rng)}")
        self.events.append({"asker": asker.name, "attr": attr.name, "option": option.id,
                            "deltas": {r.buyer: r.delta for r in reactions}, "offers": self.rnd.offers()})

    def summary(self) -> Example:
        """The whole scene, for scenes.jsonl."""
        return {"item": self.item.key, "personas": self.personas, "show_attrs": self.attrs is not None,
                "budgets": {b.name: round(b.budget, 2) for b in self.buyers}, "events": self.events,
                "log": self.log, "final_offers": self.rnd.offers()}


def simulate_scene(item: Item, traits: Traits, rng: random.Random,
                   prompt_format: int = fmt.PROMPT_FORMAT, risky: bool = False) -> tuple[Example, list[Example]]:
    """Play one round of the engine with random buyers and random true answers.

    Returns the scene (for scenes.jsonl) and its training examples: one "ask" per question and one "react" per
    buyer per answer (silent Δ = 0 reactions are kept with probability KEEP_SILENT_ZERO_P)."""
    scene = _Scene(item, traits, rng, prompt_format)
    if risky:
        scene.refuse_risky(traits)
    while (q := scene.rnd.next_question()) is not None:
        asker, attr = q
        scene.ask(asker, attr)
        scene.answer(asker, attr)
    return scene.summary(), scene.examples


def refusal_examples(noun_key: str, traits: Traits, rng: random.Random, n: int,
                     prompt_format: int = fmt.PROMPT_FORMAT) -> list[Example]:
    """n "refuse" examples for an item no buyer wants: a random buyer of one or two traits says why not."""
    out = []
    item_names = fmt.item_names(noun_key, noun_key)
    names = rng.sample(traits.names, min(n, len(traits.names)))
    for name, line in zip(names, phrases.refuse_lines(noun_key, n, rng)):
        chosen = rng.sample(list(traits.traits), rng.choice((1, 2)))
        persona = f"{name}, who " + " and ".join(traits.traits[t].persona for t in chosen) + "."
        out.append({"task": "refuse", "item": noun_key, "buyer": name, "traits": chosen,
                    "prompt": fmt.refuse_prompt(noun_key, persona),
                    "target": fmt.refuse_target(name, fmt.to_slot(line, item_names) if prompt_format >= 3 else line)})
    return out


def check(example: Example, items: dict[str, Item]) -> None:
    """Every example must parse, and its text must agree with its label."""
    task, target = example["task"], example["target"]
    if task == "react":
        parsed = fmt.parse_react(target)
        assert parsed is not None, target
        name, line, delta = parsed
        assert delta == example["delta"], (delta, example)
        assert (line is not None) == example["speaks"], example
    else:
        parsed = fmt.parse_line(target)
        assert parsed is not None, target
        assert parsed[0] == example["buyer"], example
        if task == "ask":
            item = items[example["item"]]
            question = item.attributes[example["attr"]].question
            if fmt.ITEM_SLOT in parsed[1]:
                question = fmt.to_slot(question, fmt.item_names(item.key, item.noun))
            assert question[1:-1].lower() in parsed[1].lower(), (question, parsed[1])


def split_of(i: int, heldout: bool) -> str:
    """The split of the i-th scene (or refusal) of an item: held-out items are all test_heldout; of every ten from
    a training item, the ninth goes to val, the tenth to test_seen and the rest to train."""
    if heldout:
        return "test_heldout"
    return {VAL_SLOT: "val", TEST_SLOT: "test_seen"}.get(i % SPLIT_EVERY, "train")


class _Writer:
    """The four split files, plus the counts that end up in stats.json."""

    def __init__(self, out: Path) -> None:
        self.files = {name: open(out / f"{name}.jsonl", "w") for name in SPLITS}
        self.counts: Counter[tuple[str, str]] = Counter()
        self.deltas: Counter[int] = Counter()
        self.chars = 0

    def write(self, split: str, ex: Example, items: dict[str, Item]) -> None:
        """Check the example, append it to its split's file and count it."""
        check(ex, items)
        self.files[split].write(json.dumps(ex) + "\n")
        self.counts[(split, ex["task"])] += 1
        self.chars += len(ex["prompt"]) + len(ex["target"])
        if ex["task"] == "react":
            self.deltas[ex["delta"]] += 1

    def __enter__(self) -> _Writer:
        return self

    def __exit__(self, *exc: object) -> None:
        for f in self.files.values():
            f.close()

    def stats(self, splits: dict[str, list[str]]) -> dict[str, Any]:
        """stats.json: examples per split and task, how often each Δ appears, sizes, and items per split."""
        return {"examples": {f"{s}/{t}": n for (s, t), n in sorted(self.counts.items())},
                "react_delta_counts": {f"{d:+d}": self.deltas[d] for d in sorted(self.deltas)},
                "total_examples": sum(self.counts.values()), "total_chars": self.chars,
                "items": {k: len(v) for k, v in splits.items()}}


def parse_args() -> argparse.Namespace:
    """Command-line options: scenes per item, seed, output folder and prompt format."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes-per-item", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    parser.add_argument("--prompt-format", type=int, choices=(1, 2, 3), default=fmt.PROMPT_FORMAT)
    return parser.parse_args()


def main() -> None:
    """Simulate every buy item's scenes and every refuse item's refusals, check each example, write the splits."""
    args = parse_args()
    labels, items, traits = load_labels(), load_items(), load_traits()
    splits = item_splits(labels)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    with _Writer(args.out) as writer, open(args.out / "scenes.jsonl", "w") as scene_file:
        for key in splits["buy_train"] + splits["buy_heldout"]:
            heldout = key in splits["buy_heldout"]
            for i in range(args.scenes_per_item):
                scene, examples = simulate_scene(items[key], traits, rng, args.prompt_format,
                                                 risky=labels[key] == "risky")
                scene["split"] = split = split_of(i, heldout)
                scene_file.write(json.dumps(scene) + "\n")
                for ex in examples:
                    writer.write(split, ex, items)
        for key in splits["refuse_train"] + splits["refuse_heldout"]:
            heldout = key in splits["refuse_heldout"]
            for j, ex in enumerate(refusal_examples(key, traits, rng, REFUSALS_PER_ITEM, args.prompt_format)):
                writer.write(split_of(j, heldout), ex, items)

    stats = writer.stats(splits)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    (args.out / "meta.json").write_text(json.dumps({"prompt_format": args.prompt_format, "seed": args.seed,
                                                    "scenes_per_item": args.scenes_per_item}))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
