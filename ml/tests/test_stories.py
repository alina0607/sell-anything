import json
import random
import re

from sellanything_ml.buyers import stories

# thirty distinct drawable things, three per buyer, so no two fake scenes share things
THINGS = ["couch", "pillow", "bed", "book", "cup", "chair", "vase", "flower", "lantern", "backpack", "map", "tent",
          "guitar", "piano", "clock", "basket", "bench", "candle", "teapot", "umbrella", "camera", "radio", "trumpet",
          "bicycle", "fence", "ladder", "bucket", "broom", "mug", "hat"]


def scene(k: int) -> tuple[list[str], str]:
    things = THINGS[3 * k:3 * k + 3]
    return things, f"My {things[0]}, my {things[1]} and my {things[2]} are for her."


def who(prompt: str) -> str:
    """The buyer a per-buyer prompt is for."""
    return re.search(r"You are (\w+), who", prompt).group(1)


def fake_teacher():
    """Answers each kind of prompt the way the teacher is asked to, so the turn logic can run without a model."""
    s = stories.seed_sale(random.Random(3))
    names = [b["name"] for b in s["buyers"]]

    def ask(prompt: str) -> str:
        if prompt.startswith("The buyers:"):
            return json.dumps([{"name": n, "cares_about": ["price"], "budget": 100, "says": "Hi.", "offer": 60}
                               for n in names])
        if "Choose ONE buyer" in prompt:
            left = prompt.split("has not asked a question yet (")[1].split(")")[0].split(", ")
            return json.dumps({"name": left[0], "question": "How old is she?"})
        if prompt.startswith("You are the seller"):
            return "Two years old."
        if "Picture this cat at your own place" in prompt:
            things, says = scene(names.index(who(prompt)))
            return json.dumps({"things": things, "says": says})
        return "Sure: " + json.dumps({"sign": "+", "says": "Nice.", "offer": 70})
    return s, ask


def test_a_short_sale_has_every_part_in_order():
    s, ask = fake_teacher()
    m = stories.sale(ask, s, rounds=1)
    kinds = [e["kind"] for e in m.events]
    assert kinds == ["open"] * 10 + ["ask", "answer"] + ["react"] * 10
    text = stories.render(m)
    assert text.count("<|buyer|>") == 10 and "<|round 1|>" in text and "<|answer to " in text
    assert stories.check(m) == {"imagined": 0, "things_pictured": 0, "things_not_drawable": 0, "things_not_said": 0,
                                "imagine_retried": 0, "imagine_dropped": 0, "react_retried": 0, "react_dropped": 0,
                                "reactions": 10,
                                "offer_against_sign": 0, "sign_without_move": 0, "over_budget": 0,
                                "amount_said_differs": 0, "names_self": 0, "lines_with_dash": 0}


def test_a_full_sale_lets_each_buyer_ask_once_and_ends_sold():
    s, ask = fake_teacher()
    m = stories.sale(ask, s)
    askers = [e["name"] for e in m.events if e["kind"] == "ask"]
    assert sorted(askers) == sorted(b["name"] for b in s["buyers"])
    assert m.events[-1]["kind"] == "sold"
    assert [e["kind"] for e in m.events[-11:-1]] == ["imagine"] * 10
    assert "<|imagine|> " in stories.render(m) and "things: couch, pillow, bed" in stories.render(m)
    c = stories.check(m)
    assert (c["imagined"], c["things_pictured"], c["things_not_drawable"], c["things_not_said"]) == (10, 30, 0, 0)
    assert (c["imagine_retried"], c["imagine_dropped"]) == (0, 0)


def test_a_scene_that_breaks_a_rule_is_asked_again_then_dropped():
    s, base = fake_teacher()
    names = [b["name"] for b in s["buyers"]]
    good, stubborn = names[0], names[1]

    def ask(prompt: str) -> str:
        if "Picture this cat at your own place" not in prompt:
            return base(prompt)
        n = who(prompt)
        things, says = scene(names.index(n))
        if n in (good, stubborn) and not (n == good and "broke a rule" in prompt):   # window cannot be drawn
            things, says = ["window"] + things[1:], "By the window, " + says
        return json.dumps({"things": things, "says": says})
    m = stories.sale(ask, s)
    scenes = {e["name"]: e for e in m.events if e["kind"] == "imagine"}
    assert stubborn not in scenes and scenes[good]["things"] == scene(0)[0] and len(scenes) == 9
    assert all(stories.imagine_problem(e) is None for e in scenes.values())
    c = stories.check(m)
    # rewrites asked: one for the buyer who fixed it, three for the one who never did (then dropped)
    assert (c["imagine_retried"], c["imagine_dropped"], c["things_not_drawable"]) == (stories.IMAGINE_TRIES, 1, 0)


def test_imagine_problem_names_what_is_wrong():
    assert "not on the list" in stories.imagine_problem({"things": ["window", "bed", "cup"], "says": "window"})
    assert "exactly 3" in stories.imagine_problem({"things": ["bed"], "says": "My bed."})
    three = {"things": ["alarm clock", "bed", "cup"], "says": "a clock, a bed, a cup"}
    assert stories.imagine_problem(three) is None
    assert "never names book" in stories.imagine_problem({"things": ["couch", "book", "bed"], "says": "My couch and bed."})


def test_check_counts_offers_moving_against_their_reason():
    s, ask = fake_teacher()
    m = stories.sale(ask, s, rounds=1)
    m.events[-1]["sign"] = "-"                       # says it is bad news, yet raised the offer from 60 to 70
    assert stories.check(m)["offer_against_sign"] == 1


def test_seeds_never_contradict_themselves():
    for seed in range(300):
        c = stories.seed_sale(random.Random(seed))["cat"]
        kinds = [k for k, answers in stories.COMPANY.items() for a in c["company"] if a in answers]
        assert len(kinds) == len(set(kinds)) == 2
        assert not (c["months"] < 6 and c["fixed"] == "spayed/neutered")
        assert not (c["breed"] == "a Sphynx" and "sheds a lot" in c["quirks"])
        assert not any(w in q for q in c["quirks"] for w in ("dog", "children", "cat "))


def test_a_buyer_who_withdraws_stops_asking_and_reacting():
    s, base = fake_teacher()
    quitter = s["buyers"][0]["name"]

    def ask(prompt: str) -> str:
        if "React to this latest answer only" in prompt:
            return json.dumps({"sign": "-", "says": "Not for me.", "offer": 0 if who(prompt) == quitter else 50})
        return base(prompt)
    m = stories.sale(ask, s)
    assert quitter not in m.present()
    after = m.events[next(i for i, e in enumerate(m.events) if e["kind"] == "react" and e["offer"] == 0) + 1:]
    assert all(e.get("name") != quitter for e in after)
    assert stories.check(m)["offer_against_sign"] == 0
    assert "->out" in stories.render(m)


def test_the_reviewer_reports_only_problems():
    s, base = fake_teacher()
    m = stories.sale(base, s, rounds=1)
    react = next(i for i, e in enumerate(m.events) if e["kind"] == "react")

    def ask(prompt: str) -> str:
        return json.dumps([{"id": react, "verdict": "wrong", "problem": "FIV cannot infect children"},
                           {"id": react + 1, "verdict": "ok", "problem": ""}])
    assert stories.judge(ask, m) == [{"id": react, "verdict": "wrong", "problem": "FIV cannot infect children"}]


def test_many_sales_in_one_batch_match_sales_run_one_by_one():
    s, ask = fake_teacher()
    batches = []

    def ask_many(prompts):
        batches.append(len(prompts))
        return [ask(p) for p in prompts]
    together = stories.run_many([stories.sale_steps(s, 2) for _ in range(3)], ask_many)
    alone = stories.sale(ask, s, 2)
    assert [m.events for m in together] == [alone.events] * 3
    assert batches and all(n % 3 == 0 for n in batches)   # the three sales never wait on each other


def test_knowledge_texts_come_back_with_the_reviewers_problems():
    replies = iter(["Senior cats sleep a lot.", '[{"sentence": "Senior cats sleep a lot.", "problem": "not a fact"}]'])
    out = stories.drive(stories.knowledge_steps("senior", "qa"), lambda prompt: next(replies))
    assert out["text"] == "Senior cats sleep a lot." and out["problems"][0]["problem"] == "not a fact"


def test_the_reviewer_sees_each_round_with_only_what_was_said_before_it():
    s, base = fake_teacher()
    m = stories.sale(base, s, rounds=3)
    m.events[[e["kind"] for e in m.events].index("answer", 15)]["says"] = "SECOND ANSWER"
    prompts = []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return "[]"
    assert stories.judge(ask, m) == []
    assert len(prompts) == 3
    before = [p.split("This round:")[0] for p in prompts]
    assert "SECOND ANSWER" not in before[0] + before[1] and "SECOND ANSWER" in before[2]


def test_the_reviewer_judges_only_buyers_and_never_sees_the_cats_true_facts():
    s, base = fake_teacher()
    m = stories.sale(base, s, rounds=1)
    answer = [e["kind"] for e in m.events].index("answer")
    prompts = []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return "[]"
    stories.judge(ask, m)
    assert stories.facts(s) not in prompts[0]
    assert f"[{answer}]" not in prompts[0] and m.events[answer]["says"] in prompts[0]


def test_a_reply_without_json_is_asked_again():
    replies = iter(["Senior cats sleep a lot.", "No problems found.", "[]"])
    out = stories.drive(stories.knowledge_steps("senior", "qa"), lambda prompt: next(replies))
    assert out["problems"] == []


def test_check_counts_spoken_problems():
    m = stories.Market({"cat": {}, "buyers": []})
    m.cards = {"Ruby": {"persona": "x", "cares_about": [], "budget": 100}}
    m.events = [{"kind": "react", "name": "Ruby", "old": 80, "offer": 80, "sign": "+",
                 "says": "Ruby likes her — I'll go to $90."}]
    c = stories.check(m)
    assert (c["sign_without_move"], c["amount_said_differs"], c["names_self"], c["lines_with_dash"]) == (1, 1, 1, 1)
    m.events[0].update(offer=90, sign="+")
    assert stories.check(m)["amount_said_differs"] == 0


def test_imagine_lines_name_only_the_things_picked():
    ok = {"things": ["couch", "alarm clock", "bed"], "says": "A couch for naps, an alarm clock so I never forget her, a bed."}
    assert stories.imagine_problem(ok) is None                       # "clock" is a class too, but part of "alarm clock"
    assert "mentions dog" in stories.imagine_problem({"things": ["couch", "bed", "cup"], "says": "On my couch next to the dog, bed, cup."})
    assert "window" in stories.imagine_problem({"things": ["couch", "bed", "cup"], "says": "On my couch by the window, bed, cup."})
    assert "sofa" in stories.imagine_problem({"things": ["couch", "bed", "cup"], "says": "My couch, or the sofa, bed, cup."})
    # everyday words that are also class names are not objects here
    assert stories.imagine_problem({"things": ["bed", "couch", "cup"], "says": "I'd keep an eye on her bed, couch and cup in my house."}) is None


def test_not_drawable_words_are_really_not_drawable():
    from sellanything_ml.buyers.drawable import DRAWABLE
    assert not set(stories.NOT_DRAWABLE) & set(DRAWABLE)
    assert stories.EVERYDAY_WORDS <= set(DRAWABLE)


def test_two_buyers_may_not_picture_the_same_things():
    s, base = fake_teacher()
    names = [b["name"] for b in s["buyers"]]

    def ask(prompt: str) -> str:
        if "Picture this cat at your own place" not in prompt:
            return base(prompt)
        k = names.index(who(prompt))
        if k == 1 and "broke a rule" not in prompt:      # the second buyer copies the first one's scene once
            k = 0
        return json.dumps({"things": scene(k)[0], "says": scene(k)[1]})
    m = stories.sale(ask, s)
    scenes = [e for e in m.events if e["kind"] == "imagine"]
    assert len(scenes) == 10 and stories.check(m)["imagine_retried"] == 1
    assert len({frozenset(e["things"]) for e in scenes}) == 10


def test_reaction_rules():
    card = {"budget": 100}
    ok = {"sign": "+", "offer": 80, "says": "I'm home all day, so a cat who follows me around is great. I'll go to $80."}
    assert stories.reaction_problem(ok, card, 70, "Ruby", "She follows people around.") is None
    assert "budget" in stories.reaction_problem({**ok, "offer": 120, "says": "I'll go to $120."}, card, 70, "Ruby", "")
    assert "does not match" in stories.reaction_problem({**ok, "offer": 70}, card, 70, "Ruby", "")
    assert "no reason" in stories.reaction_problem({**ok, "says": "That sounds perfect for me. $80."}, card, 70, "Ruby", "")
    keep = {"sign": "0", "offer": 70, "says": "Good to know, but I'm still waiting to hear about her health."}
    assert "no reason" in stories.reaction_problem(keep, card, 70, "Ruby", "She sheds a lot.")
    off_topic = {"sign": "0", "offer": 70, "says": "I still want an outdoorsy cat for my trips."}
    assert "name what the seller just said" in stories.reaction_problem(off_topic, card, 70, "Ines", "She is FIV positive.")
    on_topic = {"sign": "0", "offer": 70, "says": "Her shedding doesn't bother me since I vacuum daily. Still $70."}
    assert stories.reaction_problem(on_topic, card, 70, "Ines", "She sheds a lot in spring.") is None


def test_the_example_lines_follow_their_own_rules():
    for line in stories.REACTION_EXAMPLES:
        assert not any(d in line for d in stories.DASHES)
        assert not any(v in line.lower() for v in stories.VAGUE)


def test_each_buyer_reacts_from_their_own_card_only():
    s, base = fake_teacher()
    prompts = []

    def ask(prompt: str) -> str:
        if "React to this latest answer only" in prompt:
            prompts.append(prompt)
        return base(prompt)
    m = stories.sale(ask, s, rounds=2)
    names = [b["name"] for b in s["buyers"]]
    first = prompts[0]
    assert sum(f"You are {n}, who" in first for n in names) == 1        # one card per prompt
    other = next(n for n in names if n != who(first))
    assert f"{other} (" not in first.split("You are ")[-1]               # not another buyer's lines either
    assert "How old is she?" in first and sum(e["kind"] == "react" for e in m.events) >= 10


def test_she_being_is_not_english():
    card = {"budget": 100}
    row = {"sign": "0", "offer": 70, "says": "She being a tabby doesn't matter to me. Staying at $70."}
    assert "not English" in stories.reaction_problem(row, card, 70, "Ruby", "She's a tabby.")
    row["says"] = "Being a tabby doesn't matter to me. Staying at $70."
    assert stories.reaction_problem(row, card, 70, "Ruby", "She's a tabby.") is None
