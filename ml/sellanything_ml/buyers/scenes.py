"""Buyer scenes (docs/design-v2.md §5.4): "how I picture it at my place".

Rules only, no model: every prop that fits the item's group gets

    score(p) = base_p + Σ_t profile_i[t] × tags_p[t] + λ × Σ_t [profile_i[t] > 0] × weights_b[t] × tags_p[t]

where profile_i is the item's tag profile scaled so its largest tag is 2. Props with score ≤ 0 are
dropped, the top 3 are kept with at most one per zone, and each goes in its zone's box. The zone boxes
never overlap each other or the item's box, so nothing in a scene is drawn on top of something else.
The drawing GPT then draws the item and every prop (server.py, `scene`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Protocol

from .schema import DATA_DIR, Item, Traits

SCENE_W, SCENE_H = 512, 288          # scene canvas, 16:9
ITEM_BOX = (184, 104, 144, 176)      # x, y, w, h: centred and the largest box
ZONES = {                            # disjoint from each other and from ITEM_BOX (tested)
    "sky": (336, 8, 168, 88),
    "background": (8, 8, 168, 144),
    "wall": (184, 8, 144, 88),
    "beside": (336, 104, 168, 92),
    "ground": (8, 160, 168, 120),
    "ground_side": (336, 204, 168, 76),
}
SIZES = {"small": 0.55, "medium": 0.8, "large": 1.0}   # share of the zone box a prop may fill
LAMBDA = 1.0                         # weight of the buyer's taste (design §5.4: "to tune")
MAX_PROPS = 3
BOTTOM_ZONES = ("item", "ground", "ground_side", "beside")   # things that stand on something; the rest float

SceneObject = dict[str, Any]         # one thing in a scene: label, role, zone, box, share, reason (+ score, strokes)
Strokes = list[tuple[list[float], list[float]]]

# A tag a buyer cares about, in a few words (same wording as the buyer cards, rounds.REASONS).
TAG_WORDS = {"space": "fits my home", "condition": "good condition", "low_maintenance": "easy upkeep",
             "family": "good for kids", "status": "prestige", "performance": "works well",
             "style": "looks good", "active": "good outdoors"}


class SceneBuyer(Protocol):
    """What a scene needs to know about a buyer (engine.Buyer has it)."""
    traits: list[str]
    weights: dict[str, int]


@dataclass
class Prop:
    """A thing that can stand next to an item in a scene (one entry of props.json)."""
    name: str
    tags: dict[str, int]
    zone: str
    size: str
    fits: list[str] | str
    base: float = 0.0


@dataclass
class Choice:
    """A prop with its score and the reason shown for it."""
    prop: Prop
    score: float
    reason: str
    parts: dict[str, float] = field(default_factory=dict)   # base, item, buyer


def load_props(path: Path = DATA_DIR / "props.json") -> dict[str, Prop]:
    """props.json as Prop objects, by name."""
    raw = json.loads(path.read_text())["props"]
    return {name: Prop(name, p.get("tags", {}), p["zone"], p["size"], p["fits"], float(p.get("base", 0)))
            for name, p in raw.items()}


def load_groups(path: Path = DATA_DIR / "items.tsv") -> dict[str, str]:
    """Item name -> group (third column of items.tsv)."""
    rows = [line.split("\t") for line in path.read_text().splitlines()[1:] if line.strip()]
    return {r[0]: r[2] for r in rows}


@lru_cache(maxsize=1)
def tables() -> tuple[dict[str, Prop], dict[str, str]]:
    """props.json and the item groups, read once."""
    return load_props(), load_groups()


def item_profile(item: Item, answered: dict[str, str] | None = None) -> dict[str, float]:
    """§2.2: Σ over attributes and options of |signals|, per tag; an answered attribute counts its chosen
    option's signed signals instead (a senior dog, active −1, loses the grass). Scaled so the largest
    tag is 2; tags that end up ≤ 0 are left at their value (they can push props out)."""
    answered = answered or {}
    raw: dict[str, float] = {}
    for name, attr in item.attributes.items():
        if name in answered:
            signals = [attr.option(answered[name]).signals]
        else:
            signals = [{t: abs(s) for t, s in option.signals.items()} for option in attr.options]
        for one in signals:
            for t, s in one.items():
                raw[t] = raw.get(t, 0) + s
    top = max((v for v in raw.values() if v > 0), default=0)
    return {t: 2 * v / top for t, v in raw.items()} if top else {t: 0.0 for t in raw}


def score_props(item: Item, group: str, weights: dict[str, int], props: dict[str, Prop], *,
                buyer_labels: dict[str, str] | None = None, answered: dict[str, str] | None = None,
                lam: float = LAMBDA) -> list[Choice]:
    """Every prop that fits, best first. `buyer_labels` maps a tag to the label of the buyer's trait that
    weighs it most ("active" -> "Outdoorsy"), for the reason text."""
    profile = item_profile(item, answered)
    buyer_labels = buyer_labels or {}
    out = []
    for p in props.values():
        if p.name == item.key or (p.fits != "*" and group not in p.fits):
            continue
        from_item = sum(profile.get(t, 0) * v for t, v in p.tags.items())
        per_tag = {t: weights.get(t, 0) * v for t, v in p.tags.items() if profile.get(t, 0) > 0}
        from_buyer = lam * sum(per_tag.values())
        score = p.base + from_item + from_buyer
        if from_buyer > 0 and from_buyer >= max(p.base, from_item):
            tag = max(per_tag, key=per_tag.get)
            reason = f"{buyer_labels.get(tag, 'Taste')}: {TAG_WORDS[tag]}"
        else:
            reason = f"goes with a {item.noun}"
        out.append(Choice(p, round(score, 3), reason, {"base": p.base, "item": round(from_item, 3),
                                                       "buyer": round(from_buyer, 3)}))
    return sorted(out, key=lambda c: (-c.score, c.prop.name))


def choose(scored: list[Choice], k: int = MAX_PROPS) -> list[Choice]:
    """Top k with a positive score, at most one per zone."""
    picked, zones = [], set()
    for c in scored:
        if c.score <= 0 or c.prop.zone in zones:
            continue
        picked.append(c)
        zones.add(c.prop.zone)
        if len(picked) == k:
            break
    return picked


def fit(strokes: Strokes, box: tuple[float, float, float, float], share: float = 1.0,
        anchor: str = "center") -> Strokes:
    """Scale a drawing uniformly (never stretched) to fill `share` of `box`, centred horizontally;
    vertically centred, or standing on the box's bottom edge with anchor="bottom"."""
    xs = [x for s in strokes for x in s[0]]
    ys = [y for s in strokes for y in s[1]]
    if not xs:
        return []
    x0, y0, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    bx, by, bw, bh = box
    k = share * min(bw / max(w, 1), bh / max(h, 1))
    ox = bx + (bw - w * k) / 2
    oy = by + bh - h * k if anchor == "bottom" else by + (bh - h * k) / 2
    return [([round(ox + (x - x0) * k, 1) for x in sx], [round(oy + (y - y0) * k, 1) for y in sy])
            for sx, sy in strokes]


def layout(item_key: str, picked: list[Choice]) -> list[SceneObject]:
    """The item and its props with their boxes, item first (it is drawn first)."""
    objs = [{"label": item_key, "role": "item", "zone": "item", "box": list(ITEM_BOX), "share": 1.0, "reason": ""}]
    for c in picked:
        objs.append({"label": c.prop.name, "role": "prop", "zone": c.prop.zone, "box": list(ZONES[c.prop.zone]),
                     "share": SIZES[c.prop.size], "reason": c.reason, "score": c.score})
    return objs


def trait_labels(buyer: SceneBuyer, traits: Traits) -> dict[str, str]:
    """Tag -> label of the buyer's trait that weighs it most ("active" -> "Outdoorsy"); tags none of them weigh
    are left out."""
    labels: dict[str, str] = {}
    for tag in TAG_WORDS:
        best = max(buyer.traits, key=lambda t: traits.traits[t].weights.get(tag, 0), default=None)
        if best is not None and traits.traits[best].weights.get(tag, 0) > 0:
            labels[tag] = traits.traits[best].label
    return labels


def plan(item: Item, group: str, buyer: SceneBuyer, traits: Traits, props: dict[str, Prop], *,
         answered: dict[str, str] | None = None, lam: float = LAMBDA) -> list[SceneObject]:
    """The scene for one buyer of a round: objects with boxes and reasons, before anything is drawn."""
    scored = score_props(item, group, buyer.weights, props, buyer_labels=trait_labels(buyer, traits),
                         answered=answered, lam=lam)
    return layout(item.key, choose(scored))


def render(objects: list[SceneObject], draw: Callable[[str], Strokes]) -> list[SceneObject]:
    """Draw every object with `draw(label) -> strokes (0..255)` and fit it into its box."""
    out = []
    for o in objects:
        anchor = "bottom" if o["zone"] in BOTTOM_ZONES else "center"
        strokes = fit(draw(o["label"]), tuple(o["box"]), o["share"], anchor)
        out.append({**o, "strokes": [[xs, ys] for xs, ys in strokes]})
    return out
