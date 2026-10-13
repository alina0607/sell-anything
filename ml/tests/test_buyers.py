"""Price engine and host against docs/buyer-ai.md. Expected numbers are worked out by hand."""

import random

import pytest

from sellanything_ml.buyers import (Buyer, Round, interest_delta, load_items, load_labels,
                                    load_traits, make_buyers, round_price, validate)
from sellanything_ml.buyers import phrases
from sellanything_ml.buyers.schema import Attribute, Item, Option

# --- the real data files --------------------------------------------------------------------


def test_data_files_are_valid():
    assert validate(load_items(), load_traits(), load_labels()) == []


def test_every_schema_item_plays_a_full_round():
    items, traits = load_items(), load_traits()
    for key, item in items.items():
        rng = random.Random(key)
        rnd = Round(item, make_buyers(item, traits, 4, rng))
        asked = 0
        while (q := rnd.next_question()) is not None:
            buyer, attr = q
            rnd.answer(buyer, attr, rng.choice(attr.options).id)
            asked += 1
        assert asked == len(item.attributes) or not any(b.present for b in rnd.buyers), key


def test_validate_catches_mistakes():
    traits = load_traits()
    bad = Item("sun", "sun", 10, 5, 100, {})                     # refuse item, bad prices, 0 attrs
    errors = validate({"sun": bad}, traits, load_labels())
    assert any("refuse" in e for e in errors)
    assert any("low < typical < high" in e for e in errors)
    assert any("attributes" in e for e in errors)


# --- a tiny hand-made item so the arithmetic is checkable ------------------------------------

SIZE = Attribute("size", "How big?", (
    Option("small", "Small.", 1.0, {"space": 2}),
    Option("large", "Large.", 1.0, {"space": -2, "active": 1}),
))
CONDITION = Attribute("condition", "Condition?", (
    Option("mint", "Mint.", 1.5, {"condition": 2}),
    Option("broken", "Broken.", 0.5, {"condition": -2}),
))
STYLE = Attribute("style", "Looks?", (
    Option("pretty", "Pretty.", 1.0, {"style": 1}),
    Option("plain", "Plain.", 1.0, {}),
))
ITEM = Item("toy", "toy", low=10, typical=100, high=400,
            attributes={"size": SIZE, "condition": CONDITION, "style": STYLE})


def buyer(name, weights, budget=1000.0, eagerness=0.8):
    return Buyer(name, [], dict(weights), "normal", budget, eagerness)


def test_interest_delta_is_weighted_sum_clamped():
    assert interest_delta({"space": 2}, {"space": -2, "active": 1}) == -2       # raw −4 → −2
    assert interest_delta({"space": 1, "active": 2}, {"space": -2, "active": 1}) == 0
    assert interest_delta({"active": 1}, {"space": -2, "active": 1}) == 1
    assert interest_delta({}, {"space": 2}) == 0


def test_opening_offer_is_eagerness_times_willing():
    rnd = Round(ITEM, [buyer("A", {}, eagerness=0.8)])
    # nothing answered: valuation = 100 × 1 × 1 = 100, progress 0 → offer = 0.8 × 100
    assert rnd.buyers[0].offer == pytest.approx(80.0)


def test_valuation_objective_and_interest():
    a = buyer("A", {"condition": 2})
    rnd = Round(ITEM, [a])
    rnd.answer(a, CONDITION, "mint")
    # Δ = 2×2 → clamp 2; valuation = 100 × 1.5 × (1 + 0.12×2) = 186; progress 1/3
    assert a.interest == 2
    assert rnd.valuation(a) == pytest.approx(186.0)
    assert a.offer == pytest.approx(186.0 * (0.8 + 0.2 / 3))


def test_budget_caps_willingness():
    a = buyer("A", {}, budget=90.0)
    rnd = Round(ITEM, [a])
    rnd.answer(a, CONDITION, "mint")                 # valuation 150, budget 90
    assert a.willing == pytest.approx(90.0)


def test_valuation_respects_low_and_high():
    a = buyer("A", {"condition": 2})
    rnd = Round(ITEM, [a])
    a.interest = -5
    rnd.answered["condition"] = "broken"             # 100 × 0.5 × (1 − 0.6) = 20 → above low 10
    assert rnd.valuation(a) == pytest.approx(20.0)
    rnd.answered["size"] = "large"
    a.interest = 5                                   # 100 × 0.5 × 1.6 = 80
    assert rnd.valuation(a) == pytest.approx(80.0)
    rnd.answered = {"condition": "mint"}
    a.interest = 5                                   # 100 × 1.5 × 1.6 = 240 < 400
    assert rnd.valuation(a) == pytest.approx(240.0)


def test_everyone_hears_every_answer_and_walks_away():
    tiny_home = buyer("A", {"space": 2})
    hiker = buyer("B", {"active": 2})
    rnd = Round(ITEM, [tiny_home, hiker])
    reactions, _ = rnd.answer(hiker, SIZE, "large")  # B asked, but A hears it too
    deltas = {r.buyer: r.delta for r in reactions}
    assert deltas == {"A": -2, "B": 2}
    rnd.answer(hiker, CONDITION, "broken")           # A has no condition weight: still −2
    assert tiny_home.interest == -2 and tiny_home.present
    tiny_home.weights["condition"] = 2
    rnd.answered.pop("style", None)
    reactions, _ = rnd.answer(hiker, STYLE, "pretty")  # style isn't A's concern: Δ 0, stays
    assert tiny_home.present
    tiny_home.interest = -3
    rnd2 = Round(ITEM, [tiny_home, hiker])
    rnd2.answer(hiker, SIZE, "large")                 # −3 − 2 = −5 ≤ −4 → leaves
    assert not tiny_home.present and tiny_home.offer == 0.0


def test_bidding_raises_one_step_over_the_leader():
    rich = buyer("Rich", {}, budget=1000, eagerness=0.9)          # leader, no interest
    keen = buyer("Keen", {"condition": 2}, budget=1000, eagerness=0.75)
    rnd = Round(ITEM, [rich, keen])
    rnd.answer(rich, CONDITION, "mint")
    # Rich: willing 150, offer 150 × (0.9 + 0.1/3)       = 140.0
    # Keen: willing 186, offer 186 × (0.75 + 0.25/3)     = 155.0  → already leads, no raise
    assert keen.offer > rich.offer
    rich.interest = 0
    # Now make Keen trail: lower eagerness so base offer < Rich's, interest ≥ 1 → raise to best + 5
    keen2 = buyer("Keen", {"condition": 2}, budget=1000, eagerness=0.5)
    rich2 = buyer("Rich", {}, budget=1000, eagerness=0.9)
    rnd2 = Round(ITEM, [rich2, keen2])
    _, bids = rnd2.answer(rich2, CONDITION, "mint")
    best = 150 * (0.9 + 0.1 / 3)
    assert [b.buyer for b in bids] == ["Keen"]
    assert keen2.offer == pytest.approx(best + 5.0)
    # The raise is a standing bid: recomputation doesn't drop it.
    rnd2._recompute()
    assert keen2.offer == pytest.approx(best + 5.0)


def test_host_asks_the_most_concerned_buyer_and_rotates():
    tiny_home = buyer("A", {"space": 2})
    picky = buyer("B", {"condition": 2})
    rnd = Round(ITEM, [tiny_home, picky])
    b, attr = rnd.next_question()
    assert (b.name, attr.name) == ("A", "size")       # concern 2 (tie with B) → earlier buyer
    rnd.answer(b, attr, "small")
    b, attr = rnd.next_question()
    assert (b.name, attr.name) == ("B", "condition")  # A just asked
    rnd.answer(b, attr, "mint")
    b, attr = rnd.next_question()
    assert attr.name == "style"                       # nobody cares → fewest asked, first unasked
    rnd.answer(b, attr, "plain")
    assert rnd.finished()


def test_reactions_asker_plus_at_most_one_more():
    people = [buyer(n, {"space": 2}) for n in "ABCD"]
    rnd = Round(ITEM, people)
    reactions, _ = rnd.answer(people[2], SIZE, "small")
    speakers = [r.buyer for r in reactions if r.speaks]
    assert "C" in speakers and len(speakers) == 2


@pytest.mark.parametrize("x, expected", [
    (3.1, 3.0), (3.13, 3.25), (42.4, 42.0), (437.0, 435.0), (438.0, 440.0), (1234.0, 1225.0),
])
def test_round_price(x, expected):
    assert round_price(x) == expected


def test_make_buyers_is_deterministic_and_distinct():
    items, traits = load_items(), load_traits()
    a = make_buyers(items["dog"], traits, 5, random.Random(7))
    b = make_buyers(items["dog"], traits, 5, random.Random(7))
    assert [x.name for x in a] == [x.name for x in b]
    assert len({x.name for x in a}) == 5
    assert all(set(x.traits) for x in a)
    for x in a:
        lo, hi = traits.budgets[x.budget_level]
        assert lo * 200 <= x.budget <= hi * 200


def test_no_buyer_has_two_budget_setting_traits():
    items, traits = load_items(), load_traits()
    for seed in range(200):
        for b in make_buyers(items["bicycle"], traits, 5, random.Random(seed)):
            assert sum(1 for t in b.traits if traits.traits[t].budget) <= 1, b.traits


def test_one_of_a_kind_things_take_the():
    assert phrases.with_article("sun") == "the sun" and phrases.with_article("tiger") == "a tiger"


def test_condition_reasons_fit_what_the_answer_is_about():
    import random

    from sellanything_ml.buyers import phrases
    assert phrases.topic("health", "pets_livestock", ["age", "health"]) == "health"
    assert phrases.topic("health", "plants", ["type", "health"]) == "plant_health"
    assert phrases.topic("freshness", "food", ["freshness"]) == "fresh"
    assert phrases.topic("age", "risky", ["age", "temperament"]) == "animal_age"     # a tiger
    assert phrases.topic("condition", "risky", ["condition"]) is None                # a sword
    assert phrases.topic("litter", "pets_livestock", ["litter"]) is None             # training, not health
    assert phrases.topic("age", "furniture", ["age"]) is None                        # an old chair: wear
    rng = random.Random(0)
    lines = {phrases.react_with_reason(-1, "cat", 1.0, "condition", rng, "health") for _ in range(200)}
    assert not any("wear" in line or "shape" in line for line in lines)
    assert any(phrases.react_reasons_in(line) == {("condition", -1)} for line in lines)   # the guardrail still reads them
