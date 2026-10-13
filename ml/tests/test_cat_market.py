"""The game's cat sale run by the buyer GPT (buyers/cat_market.py), with a scripted stand-in for the model."""

from sellanything_ml.buyers import cat_market as C
from sellanything_ml.buyers.format import ASK, IMAGINE, OPEN, REACT

CARDS = {p: [{"cares_about": ["calm"], "budget": 80}] for p in C.stories.PERSONAS}


class ScriptedGPT:
    """Answers each task with a fixed line in the trained shape, and records the prompts it was given."""

    prompt_format = 5
    name = "buyer_gpt_scripted"

    def __init__(self, react: str = "+ | Calm is good for me. Up to $60. | $60"):
        self.prompts: list[str] = []
        self.react = react

    def say(self, prompt: str, task: str, seed: int) -> str:
        self.prompts.append(prompt)
        return {"open": "Hi there. | $50", "ask": "Is she calm?", "react": self.react,
                "imagine": "chair, sun, book | She'd nap on my chair in the sun while I read my book."}[task]


def test_parse_reads_the_trained_shapes_and_nothing_else():
    assert C.parse("open", "Hello. | $55") == {"says": "Hello.", "offer": 55}
    assert C.parse("react", "- | No papers rules her out. | out") == {"sign": "-", "says": "No papers rules her out.",
                                                                     "offer": 0}
    assert C.parse("react", "+ | Nice.") is None and C.parse("open", "Hello") is None
    assert C.parse("imagine", "dog, sun, bench | text")["things"] == ["dog", "sun", "bench"]


def test_a_sale_runs_open_ask_react_and_sell():
    gpt = ScriptedGPT()
    m = C.CatMarket.start(gpt, 3, seed=1, cards=CARDS)
    assert m.question is not None and m.state()["question"]["options"] == []
    assert all(OPEN in p for p in gpt.prompts[:3]) and ASK in gpt.prompts[3]
    for _ in range(3):
        lines, heard = m.answer_text("She is very calm.")
        assert heard["answer"] == "She is very calm." and lines[0] == "You: She is very calm."
    assert m.question is None and m.state()["finished"]
    reacts = [p for p in gpt.prompts if p.endswith(REACT)]
    assert len(reacts) == 9 and "Seller: She is very calm." in reacts[0]
    best = max(m.present(), key=m.offers.get)
    assert m.sell(best)["price"] == 60 and m.moods[best] == "won"
    objects = m.scene(best)
    assert [o["label"] for o in objects] == ["cat", "chair", "sun", "book"] and gpt.prompts[-1].endswith(IMAGINE)
    assert m.scene(best) is objects                      # written once


def test_offers_are_kept_as_written_and_unreadable_lines_are_counted_not_replaced():
    gpt = ScriptedGPT(react="+ | Up to $500. | $500")
    m = C.CatMarket.start(gpt, 2, seed=2, cards=CARDS)
    m.answer_text("Yes.")
    assert all(m.offers[n] == 500 for n in m.present())          # no spending limit (owner, 2026-10-12)
    assert "budget" not in gpt.prompts[0]
    m = C.CatMarket.start(ScriptedGPT(react="I like her"), 2, seed=2, cards=CARDS)
    m.answer_text("Yes.")
    assert all(m.offers[n] == 50 for n in m.cards) and m.stats["unreadable"] == 2
    assert m.sources.count("model_unreadable") == 2


def test_a_format_4_model_only_gets_cards_with_a_budget():
    gpt = ScriptedGPT()
    gpt.prompt_format = 4
    pool = {p: [{"cares_about": ["calm"]}, {"cares_about": ["quiet"], "budget": 70}] for p in C.stories.PERSONAS}
    m = C.CatMarket.start(gpt, 3, seed=4, cards=pool)
    assert all(c.get("budget") == 70 for c in m.cards.values()) and "budget $70" in gpt.prompts[0]
