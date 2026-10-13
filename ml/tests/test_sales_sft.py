"""B-plan SFT examples from cat sales (buyers/sales_sft.py)."""

from sellanything_ml.buyers import format as fmt
from sellanything_ml.buyers import sales_sft as S

CARDS = {"Ann": {"persona": "has two dogs", "cares_about": ["used to dogs"], "budget": 90},
         "Bo": {"persona": "is a student", "cares_about": ["cheap"], "budget": 40}}


def sale(rounds: int = 3) -> dict:
    events = [{"kind": "open", "name": "Ann", "offer": 60, "says": "Hi, I have dogs."},
              {"kind": "open", "name": "Bo", "offer": 30, "says": "Hey, money is tight."}]
    offers = {"Ann": 60, "Bo": 30}
    for k in range(1, rounds + 1):
        asker = "Ann" if k % 2 else "Bo"
        events += [{"kind": "ask", "round": k, "name": asker, "offer": 1, "says": f"Question {k}?"},
                   {"kind": "answer", "round": k, "to": asker, "q": f"Question {k}?", "says": f"Answer {k}."}]
        for n in ("Ann", "Bo"):
            new = offers[n] + 5
            events.append({"kind": "react", "name": n, "old": offers[n], "offer": new, "sign": "+",
                           "says": f"{n} likes answer {k}. Up to ${new}."})
            offers[n] = new
    events += [{"kind": "imagine", "name": "Ann", "things": ["dog", "bench", "sun"], "says": "My dog, a bench, sun."},
               {"kind": "sold", "name": "Ann", "price": offers["Ann"]}]
    return {"seed": 7, "cards": CARDS, "events": events}


def test_one_example_per_decision_with_the_offer_from_the_reactions():
    ex = S.examples(sale())
    assert [e["task"] for e in ex].count("react") == 6 and [e["task"] for e in ex].count("ask") == 3
    ask3 = [e for e in ex if e["task"] == "ask"][2]
    assert "offer now $70" in ask3["prompt"]                 # 60 + 5 + 5, not the stale offer in the ask event
    assert ex[0]["target"] == "Hi, I have dogs. | $60" + fmt.END and fmt.OPEN in ex[0]["prompt"]


def test_a_buyer_sees_only_their_own_reactions_and_the_last_k_rounds():
    ex = S.examples(sale(4))
    last_ann = [e for e in ex if e["task"] == "react" and e["name"] == "Ann"][-1]["prompt"]
    assert "Bo likes" not in last_ann and "Bo asked: Question 4?" in last_ann
    assert "Asked earlier: Question 1?\n" in last_ann                # rounds 2 and 3 in full, then round 4 so far
    assert "Answer 1." not in last_ann and "Answer 2." in last_ann and "Answer 4." in last_ann
    assert "You ($70 -> $75): Ann likes answer 3." in last_ann
    assert last_ann.endswith(fmt.REACT)


def test_react_and_imagine_targets():
    ex = S.examples(sale())
    react = [e for e in ex if e["task"] == "react"][0]
    assert react["target"] == "+ | Ann likes answer 1. Up to $65. | $65" + fmt.END
    imagine = [e for e in ex if e["task"] == "imagine"][0]
    assert imagine["target"].startswith("dog, bench, sun | ") and imagine["prompt"].endswith(fmt.IMAGINE)


def test_withdrawal_is_out_and_the_split_is_stable():
    assert S.money(0) == "out" and S.money(45) == "$45"
    assert [S.is_val(s) for s in range(2000, 2100)] == [S.is_val(s) for s in range(2000, 2100)]
    assert 3 <= sum(S.is_val(s) for s in range(2000, 2100)) <= 20


def test_only_format_4_shows_a_budget():
    c = CARDS["Ann"]
    assert "budget $90" in S.card("Ann", c, 60, prompt_format=4)
    assert "budget" not in S.card("Ann", c, 60) and S.card("Ann", c, 60).endswith("offer now $60")


def test_new_task_tokens_keep_their_ids():
    assert fmt.EXTRA_TOKENS.index(fmt.OPEN) == 6 and fmt.EXTRA_TOKENS.index(fmt.IMAGINE) == 7
    assert len(fmt.EXTRA_TOKENS) == 14
