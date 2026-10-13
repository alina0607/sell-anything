"""Price engine and host. Implements docs/buyer-ai.md §3–6 exactly; section numbers are cited inline."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .schema import Attribute, Item, Traits

INTEREST_STEP = 0.12      # §4: each point of interest moves valuation by 12%
INTEREST_LIMIT = 5        # §4: interest is clamped to [-5, +5]
WALK_AWAY_AT = -4         # §4
BID_STEP = 0.05           # §5: raise by 5% of typical
MAX_SPEAKERS = 2          # §6.5
EAGERNESS_RANGE = (0.75, 0.9)   # §4: share of what a buyer is willing to pay that they offer at the start


def round_price(x: float) -> float:
    """§4 rounding table."""
    if x < 10:
        step = 0.25
    elif x < 100:
        step = 1
    elif x < 1000:
        step = 5
    else:
        step = 25
    return round(round(x / step) * step, 2)


def interest_delta(weights: dict[str, int], signals: dict[str, int]) -> int:
    """§4: Δ = clamp(Σ weights[tag] × signals[tag], −2, +2)."""
    raw = sum(weights.get(tag, 0) * s for tag, s in signals.items())
    return max(-2, min(2, raw))


@dataclass
class Buyer:
    """One buyer in a round: who they are (traits, tag weights, budget) and where they stand (interest, offer)."""
    name: str
    traits: list[str]
    weights: dict[str, int]
    budget_level: str
    budget: float
    eagerness: float
    interest: int = 0
    present: bool = True
    asked: int = 0
    raised_to: float = 0.0    # standing bid from §5; survives recomputation
    offer: float = 0.0
    willing: float = 0.0

    def persona(self, traits: Traits) -> str:
        """'Maya, who lives in a small apartment and wants things in excellent shape.'"""
        parts = [traits.traits[t].persona for t in self.traits]
        return f"{self.name}, who " + " and ".join(parts) + "."


def refuses_risky(buyer: Buyer, traits: Traits) -> str | None:
    """For a 'risky' item: the trait that makes this buyer walk away at once, or None if they stay."""
    return next((t for t in buyer.traits if traits.traits[t].refuses_risky), None)


def make_buyers(item: Item, traits: Traits, n: int, rng: random.Random) -> list[Buyer]:
    """§3: n buyers with distinct names and, where possible, distinct traits."""
    names = rng.sample(traits.names, n)
    pool = list(traits.traits)
    rng.shuffle(pool)
    buyers = []
    for i in range(n):
        k = rng.choice((1, 2))
        chosen = [pool.pop() for _ in range(min(k, len(pool)))] or [rng.choice(list(traits.traits))]
        # "loves premium things" + "on a tight budget" can't both set the budget: keep the first.
        with_budget = [t for t in chosen if traits.traits[t].budget]
        if len(with_budget) > 1:
            chosen.remove(with_budget[1])
        weights: dict[str, int] = {}
        for t in chosen:
            for tag, w in traits.traits[t].weights.items():
                weights[tag] = min(2, weights.get(tag, 0) + w)
        level = next((traits.traits[t].budget for t in chosen if traits.traits[t].budget), None)
        if level is None:
            levels, odds = zip(*traits.default_budget_odds.items())
            level = rng.choices(levels, odds)[0]
        lo, hi = traits.budgets[level]
        budget = item.typical * rng.uniform(lo, hi)
        buyers.append(Buyer(names[i], chosen, weights, level, budget, rng.uniform(*EAGERNESS_RANGE)))
    return buyers


@dataclass
class Reaction:
    """How one buyer took an answer: the interest change Δ, whether they say something, whether they left."""
    buyer: str
    delta: int
    speaks: bool
    left: bool          # interest hit the walk-away threshold on this answer


@dataclass
class Bid:
    """A buyer raising their offer to outbid the leader (§5)."""
    buyer: str
    amount: float


@dataclass
class Round:
    """One item being sold: the buyers, the answers so far, and the host that picks who asks what next."""
    item: Item
    buyers: list[Buyer]
    answered: dict[str, str] = field(default_factory=dict)   # attribute -> option id
    last_asker: str | None = None

    def __post_init__(self) -> None:
        self._recompute()

    # --- §4 valuation ------------------------------------------------------------------

    @property
    def objective(self) -> float:
        """§4: the product of the price factors of every answer so far (1.0 before any answer)."""
        value = 1.0
        for name, option_id in self.answered.items():
            value *= self.item.attributes[name].option(option_id).factor
        return value

    @property
    def progress(self) -> float:
        """Share of the item's attributes answered so far, 0..1."""
        return len(self.answered) / len(self.item.attributes)

    def valuation(self, buyer: Buyer) -> float:
        """§4: what the item is worth to this buyer, typical × objective × interest, kept within low..high."""
        raw = self.item.typical * self.objective * (1 + INTEREST_STEP * buyer.interest)
        return max(self.item.low, min(self.item.high, raw))

    def dismiss(self, buyer: Buyer) -> None:
        """A buyer leaves before the questions start (a risky item they won't touch)."""
        buyer.present = False
        self._recompute()

    def _recompute(self) -> None:
        """§4: every buyer's willingness (valuation capped by budget) and offer, which rises from eagerness × willing
        to willing as the questions run out and never drops below a standing bid; 0 for a buyer who left."""
        for b in self.buyers:
            b.willing = min(self.valuation(b), b.budget)
            base = b.willing * (b.eagerness + (1 - b.eagerness) * self.progress)
            b.offer = min(b.willing, max(base, b.raised_to)) if b.present else 0.0

    # --- §5 bidding --------------------------------------------------------------------

    def _bid(self) -> list[Bid]:
        """§5: interested buyers (interest ≥ 1), most interested first, outbid the leader by one step while they
        are willing to pay that much."""
        present = [b for b in self.buyers if b.present]
        if len(present) < 2:
            return []
        step = BID_STEP * self.item.typical
        bids = []
        leader = max(present, key=lambda b: b.offer)
        best = leader.offer
        for b in sorted(present, key=lambda b: (-b.interest, b.name)):
            if b is leader or b.interest < 1:
                continue
            if b.offer < best and b.willing >= best + step:
                b.raised_to = best + step
                b.offer = b.raised_to
                best = b.offer
                leader = b
                bids.append(Bid(b.name, b.offer))
        return bids

    # --- §6 host -----------------------------------------------------------------------

    def unasked(self) -> list[Attribute]:
        """Attributes nobody has asked about yet, in the item's order."""
        return [a for name, a in self.item.attributes.items() if name not in self.answered]

    @staticmethod
    def concern(buyer: Buyer, attr: Attribute) -> int:
        """§6: how much the buyer cares about an attribute, the sum of their weights on the tags it moves."""
        return sum(buyer.weights.get(tag, 0) for tag in attr.tags)

    def next_question(self) -> tuple[Buyer, Attribute] | None:
        """§6: the buyer and attribute with the most concern, preferring someone other than the last asker, then
        whoever asked least; None when nobody is left or everything is answered."""
        present = [b for b in self.buyers if b.present]
        remaining = self.unasked()
        if not present or not remaining:
            return None
        order = {name: i for i, name in enumerate(self.item.attributes)}
        candidates = [(self.concern(b, a), b, a) for b in present for a in remaining]
        if max(c for c, _, _ in candidates) == 0:
            asker = min(present, key=lambda b: (b.asked, self.buyers.index(b)))
            return asker, remaining[0]
        others_care = any(c > 0 and b.name != self.last_asker for c, b, _ in candidates)
        if others_care:
            candidates = [t for t in candidates if t[1].name != self.last_asker]
        _, buyer, attr = max(
            candidates,
            key=lambda t: (t[0], -t[1].asked, -self.buyers.index(t[1]), -order[t[2].name]))
        return buyer, attr

    def answer(self, asker: Buyer, attr: Attribute, option_id: str) -> tuple[list[Reaction], list[Bid]]:
        """Apply the player's answer to every present buyer (§6.4) and run bidding (§5)."""
        option = attr.option(option_id)
        self.answered[attr.name] = option_id
        asker.asked += 1
        self.last_asker = asker.name

        reactions = []
        for b in self.buyers:
            if not b.present:
                continue
            d = interest_delta(b.weights, option.signals)
            b.interest = max(-INTEREST_LIMIT, min(INTEREST_LIMIT, b.interest + d))
            left = b.interest <= WALK_AWAY_AT
            if left:
                b.present = False
            reactions.append(Reaction(b.name, d, speaks=False, left=left))

        # §6.5: the asker always reacts; then others with |Δ| ≥ 1, largest first, up to two in total.
        by_name = {r.buyer: r for r in reactions}
        if asker.name in by_name:
            by_name[asker.name].speaks = True
        others = sorted((r for r in reactions if r.buyer != asker.name and (abs(r.delta) >= 1 or r.left)),
                        key=lambda r: (-abs(r.delta), r.buyer))
        for r in others[:MAX_SPEAKERS - 1]:
            r.speaks = True

        self._recompute()
        bids = self._bid()
        return reactions, bids

    def finished(self) -> bool:
        """No question is left to ask."""
        return self.next_question() is None

    def offers(self) -> dict[str, float]:
        """Displayed (rounded) offers of buyers still in the round."""
        return {b.name: round_price(b.offer) for b in self.buyers if b.present}
