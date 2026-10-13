"""Cat sales written turn by turn by a teacher model, TinyStories-style (Eldan & Li, 2023).

The program only decides whose turn it is; the teacher writes every line. A sale is:

- an opening: the teacher writes each buyer's card (what they care about in a cat, the most they can pay), an
  opening line and a starting offer;
- rounds: one buyer who has not asked yet asks what matters to them, the seller answers that buyer from the cat's
  true facts, and every buyer reacts: first WHY (which item on their own card the answer touches, good or bad,
  with what it means for a cat), then what they say, then their new offer;
- after the last round, the seller takes the best offer.

`render` lays a sale out with explicit markers (each round, who asked, whom the answer is to, every reaction with
its old and new offer), so a model trained on it can see which answer each reaction responds to. `check` counts
reactions whose offer moves against their own WHY, or past the buyer's budget.

Facts about the cat and the buyers come from random seeds, so every sale differs. The teacher is
Qwen2.5-32B-Instruct (Apache-2.0, so its text may train other models): on this Mac through MLX for checking the
format one sale at a time, and on Kaggle's GPUs through vLLM (tools/kaggle_dialogues/) for volume.

    python -m sellanything_ml.buyers.stories --seed 1 --rounds 1      # opening + one question, on the Mac
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, TypeVar

from .cat_knowledge import HELD_OUT, TOPICS, knowledge_prompt
from .drawable import DRAWABLE

ML_DIR = Path(__file__).resolve().parents[2]
LOCAL_TEACHER = ML_DIR / "data" / "raw" / "teacher" / "qwen2.5-32b-instruct-4bit"
OUT_DIR = ML_DIR / "runs" / "buyer_dialogues"
N_BUYERS = 10
JSON_TRIES = 3

AGES_MONTHS = [2, 4, 7, 12, 24, 36, 60, 72, 96, 120, 156, 180]
BREEDS = ["a mixed-breed tabby", "a black domestic shorthair", "a Ragdoll", "a Maine Coon", "a Siamese",
          "a British Shorthair", "a Persian", "a Bengal", "a Sphynx", "a calico", "a Scottish Fold",
          "an orange tabby of unknown breed"]
HOME = ["indoor-only", "indoor with a catio", "indoor-outdoor", "used to roaming a farm"]
HEALTH_KITTEN = ["no vaccines yet", "had the first of three kitten vaccines", "had two of three kitten vaccines",
                 "finished the kitten vaccine series", "dewormed but not vaccinated"]
HEALTH_ADULT = ["fully vaccinated, vet check last month", "vaccinated but no vet visit in two years",
                "no vaccination records", "recovering from a cold", "microchipped and healthy"]
HEALTH_SENIOR = ["bloodwork last month was normal", "bloodwork shows early kidney disease, on a kidney diet",
                 "no bloodwork in years", "on medication for an overactive thyroid", "has some stiff joints"]
VIRUS = ["tested negative for FIV and FeLV", "never tested for FIV or FeLV", "tested FIV positive"]
COMPANY = {"dogs": ["has lived with a dog", "has never met a dog"],          # one answer per kind of company
           "cats": ["has lived with another cat", "has never met another cat"],
           "children": ["is fine with children", "has never met children"]}
TEMPER = ["very cuddly", "independent", "shy with strangers", "playful and noisy at night", "calm and lazy",
          "a bit feisty, scratches when annoyed", "follows people around like a dog"]
QUIRKS = ["only drinks from a running tap", "uses a scratching post, never the sofa", "sheds a lot",
          "meows loudly for breakfast", "picky about litter", "afraid of the vacuum"]   # company is in COMPANY
PERSONAS = ["lives in a small apartment", "wants things in excellent shape", "works long hours and has little free time",
            "is buying for a family with young kids", "loves rare, premium and pedigree things", "mostly wants a good deal",
            "spends weekends hiking and wants an adventure cat", "cares a lot about how things look",
            "is a student on a tight budget", "already has two dogs", "is allergic and worried about fur",
            "is retired and home all day", "lost an old cat last year", "lives on a busy road", "breeds show cats",
            "is a nurse working night shifts", "already has an older cat at home"]
NAMES = ["Maya", "Owen", "Felix", "Elena", "Aisha", "Diego", "Andre", "Chloe", "Hana", "Ben", "Jonah", "Ruby",
         "Leo", "Mei", "Sam", "Priya", "Tomas", "Ines"]
STYLES = ["short and sometimes vague", "friendly and detailed", "makes typos and uses abbreviations",
          "honest even about the cat's flaws", "tries to oversell the cat"]
SETTING = ("A secondhand marketplace. A seller is selling a cat; a typical price for a cat here is about $75. "
           "Buyers speak naturally and in character, in one or two sentences, and know real things about cats. "
           "Everything anyone says is plain spoken English, the way people talk face to face: never use a dash "
           "(— or –) in a spoken line.")
DASHES = ("—", "–")

Ask = Callable[[str], str]           # prompt → the teacher's reply
T = TypeVar("T")
Steps = Generator[Any, Any, T]       # yields a prompt (or a list of prompts, one per buyer) for the teacher, is
                                     # sent its reply (or the list of replies), returns T
Sale = dict[str, Any]                # cat facts, buyers, seller style, typical price (seed_sale)
Event = dict[str, Any]               # one step of a sale: open, ask, answer, react or sold
JSON_SPAN = re.compile(r"[\[{].*[\]}]", re.S)
SIGNS = {"+": 1, "-": -1, "0": 0}


def _age(months: int) -> str:
    """The age as a seller says it: "4 months", "1 year", "13 years"."""
    return f"{months} months" if months < 12 else f"{months // 12} year{'s' if months >= 24 else ''}"


def seed_sale(rng: random.Random) -> Sale:
    """Random but possible facts: kittens under 6 months are not fixed yet; a Sphynx has no fur to shed."""
    months, breed = rng.choice(AGES_MONTHS), rng.choice(BREEDS)
    fixed = "not yet spayed/neutered" if months < 6 else rng.choice(["spayed/neutered", "not yet spayed/neutered"])
    quirks = [q for q in QUIRKS if not (breed == "a Sphynx" and q == "sheds a lot")]
    health = HEALTH_KITTEN if months < 12 else HEALTH_SENIOR if months > 120 else HEALTH_ADULT
    cat = {"age": _age(months), "months": months, "breed": breed, "home": rng.choice(HOME), "fixed": fixed,
           "health": rng.choice(health), "virus": rng.choice(VIRUS),
           "company": [rng.choice(COMPANY[k]) for k in rng.sample(sorted(COMPANY), 2)],
           "temperament": rng.choice(TEMPER), "quirks": rng.sample(quirks, 2)}
    buyers = [{"name": n, "persona": p}
              for n, p in zip(rng.sample(NAMES, N_BUYERS), rng.sample(PERSONAS, N_BUYERS), strict=True)]
    return {"cat": cat, "buyers": buyers, "seller_style": rng.choice(STYLES), "typical_price": 75}


def facts(s: Sale) -> str:
    """The cat's true facts in one line, as the seller is told them."""
    c = s["cat"]
    return (f"{c['age']} old, {c['breed']}, {c['home']}, {c['fixed']}, {c['health']}, {c['virus']}, "
            f"{c['company'][0]}, {c['company'][1]}, temperament: {c['temperament']}; also: {c['quirks'][0]}; "
            f"{c['quirks'][1]}")


BREED_TOPICS = {"a Ragdoll": "breed_hcm", "a Maine Coon": "breed_hcm", "a Persian": "breed_pkd",
                "a Scottish Fold": "breed_fold"}
PERSONA_TOPICS = {"dogs": "dogs", "kids": "children", "allergic": "allergies", "busy road": "outdoors"}


def _stage(months: int) -> str:
    """The life-stage topic for an age in months (cat_knowledge.TOPICS)."""
    return "kitten" if months < 12 else "young_adult" if months < 84 else "mature_adult" if months <= 120 else "senior"


def topics_for(s: Sale) -> list[str]:
    """The knowledge topics this sale touches (by the cat's age and breed, its home, and who the buyers are);
    topics held out of dialogues (cat_knowledge.HELD_OUT) never are."""
    c = s["cat"]
    personas = " ".join(b["persona"] for b in s["buyers"])
    found = [_stage(c["months"]), "other_cats", "care", BREED_TOPICS.get(c["breed"]),
             "outdoors" if c["home"] != "indoor-only" else None]
    found += [topic for word, topic in PERSONA_TOPICS.items() if word in personas]
    held = set(HELD_OUT)
    return [t for t in dict.fromkeys(found) if t and t not in held]


def knowledge(s: Sale) -> str:
    """The general cat facts for the topics this sale touches, one per line."""
    return "\n".join(f"- {f}" for t in topics_for(s) for f in TOPICS[t]["facts"])


def _json(prompt: str) -> Steps[object]:
    """The teacher's reply parsed as JSON (the outermost list or object in it), retried JSON_TRIES times."""
    for _ in range(JSON_TRIES):
        m = JSON_SPAN.search((yield prompt + "\n\nReply with JSON only, no other text."))
        if m is None:
            continue
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def _json_each(prompts: list[str]) -> Steps[list[object]]:
    """Several prompts sent together (one per buyer), each reply parsed as JSON; a reply without JSON is asked again,
    up to JSON_TRIES times, and is None if it never has any."""
    out: list[object] = [None] * len(prompts)
    todo = list(range(len(prompts)))
    for _ in range(JSON_TRIES):
        if not todo:
            break
        replies = yield [prompts[i] + "\n\nReply with JSON only, no other text." for i in todo]
        again = []
        for i, reply in zip(todo, replies, strict=True):
            m = JSON_SPAN.search(reply)
            try:
                out[i] = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                out[i] = None
            if out[i] is None:
                again.append(i)
        todo = again
    return out


def _offer_move(e: Event) -> str:
    """"$70 -> $80" for a reaction, "$70" for an opening or a question."""
    return f"${e['old']} -> ${e['offer']}" if "old" in e else f"${e['offer']}"


class Market:
    """The buyers' cards and offers, and every step of the sale so far."""

    def __init__(self, s: Sale):
        self.s = s
        self.cards: dict[str, dict[str, Any]] = {}
        self.offers: dict[str, int] = {}
        self.events: list[Event] = []
        self.react_retried = 0           # rewrites asked (buyer × rewrite) for reactions that broke a rule
        self.react_dropped = 0           # reactions still breaking a rule after the rewrites, left out
        self.imagine_retried = 0         # rewrites asked (buyer × rewrite) for scenes that broke a rule
        self.imagine_dropped = 0         # scenes still breaking a rule after the rewrites, left out

    def present(self) -> list[str]:
        """Buyers still bidding: a buyer who withdrew (offer 0) no longer asks or reacts."""
        return [n for n in self.cards if self.offers[n] > 0]

    def card_text(self) -> str:
        """Every bidding buyer's card and current offer, one per line (the prompt that picks the next question)."""
        return "\n".join(f"- {n}, who {self.cards[n]['persona']}; cares about: "
                         f"{', '.join(self.cards[n]['cares_about'])}; can pay at most ${self.cards[n]['budget']}; "
                         f"current offer ${self.offers[n]}" for n in self.present())

    def questions_and_answers(self, upto: int | None = None) -> str:
        """Every question asked so far and the seller's answer, oldest first: what every buyer heard."""
        return "\n".join(f"{e['to']} asked: {e['q']}\nSELLER: {e['says']}" for e in self.events[:upto]
                         if e["kind"] == "answer") or "(no questions yet)"

    def own_lines(self, name: str) -> str:
        """What one buyer has said so far (opening, question, reactions), with their offers: their own side of the sale,
        without the other buyers' lines (a buyer bids from their own card; owner, 2026-10-11)."""
        label = {"open": "Opening", "ask": "Your question", "react": "Your reaction"}
        mine = [e for e in self.events if e.get("name") == name and e["kind"] in label]     # not answers or scenes
        return "\n".join(f"{label[e['kind']]} ({_offer_move(e)}): {e['says']}" for e in mine) or "(nothing yet)"

    def card_line(self, name: str) -> str:
        """One buyer's own card and offer, addressed to them ("You are Maya, who ...")."""
        c = self.cards[name]
        return (f"You are {name}, who {c['persona']}. You care about: {', '.join(c['cares_about'])}. You can pay at "
                f"most ${c['budget']}. Your offer now: ${self.offers[name]}.")

    def transcript(self, upto: int | None = None) -> str:
        """The sale as the buyers heard it, up to (not including) event `upto`."""
        out = []
        for e in self.events[:upto]:
            if e["kind"] in ("open", "ask"):
                out.append(f"{e['name']} (${e['offer']}): {e['says']}")
            elif e["kind"] == "answer":
                out.append(f"SELLER to {e['to']}: {e['says']}")
            elif e["kind"] == "react":
                new = f"${e['offer']}" if e["offer"] else "withdraws"
                out.append(f"{e['name']} (${e['old']} -> {new}): {e['says']}")
            elif e["kind"] == "sold":
                out.append(f"SOLD to {e['name']} for ${e['price']}")
            elif e["kind"] == "imagine":
                out.append(f"{e['name']} pictures it with {', '.join(e['things'])}: {e['says']}")
            else:
                raise ValueError(f"unknown event kind {e['kind']!r}")
        return "\n".join(out)


def _opening(m: Market) -> Steps[None]:
    """The teacher writes every buyer's card, greeting and starting offer in one reply; buyers it leaves out or
    writes without a budget, offer or greeting are not in the sale."""
    rows = yield from _json("The buyers:\n" + "\n".join(f"- {b['name']}, who {b['persona']}" for b in m.s["buyers"]) +
                 "\n\nFor each buyer, write their card and their opening: what they care about in a cat (two or three "
                 "short phrases that follow from who they are), the most they can pay, a greeting, and a starting "
                 "offer no higher than that. The greeting tells one concrete thing about their own life that explains "
                 "what they want (not the formula 'I am X so I want Y'), and the ten greetings differ in length and "
                 'tone. Return a JSON list, one object per buyer in this order: '
                 '{"name": ..., "cares_about": [...], "budget": <dollars>, "says": ..., "offer": <dollars>}.')
    personas = {b["name"]: b["persona"] for b in m.s["buyers"]}
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("name") in personas and {"budget", "offer", "says"} <= row.keys():
            name = row["name"]
            m.cards[name] = {"persona": personas[name], "cares_about": list(row.get("cares_about", [])),
                             "budget": int(row["budget"])}
            m.offers[name] = int(row["offer"])
            m.events.append({"kind": "open", "name": name, "offer": m.offers[name], "says": row["says"]})


def _answer(m: Market, name: str, question: str) -> Steps[str]:
    """The seller sees the cat's facts, what is true about cats in general, its own earlier answers and only the
    buyer now asking. What the seller says about this cat is not judged (owner); what it says about cats in general
    must agree with the facts (owner, 2026-10-11: no "FIV can pass to dogs")."""
    earlier = "\n".join(f"Q ({e['to']}): {e['q']}\nA: {e['says']}" for e in m.events if e["kind"] == "answer")
    return (yield f"You are the seller. True facts about your cat: {facts(m.s)}.\nYour way of answering: "
               f"{m.s['seller_style']}.\n\nTrue facts about cats in general:\n{knowledge(m.s)}\n\n"
               f"Your earlier answers:\n{earlier or '(none yet)'}\n\n{name} asks: "
               f"{question}\n\nAnswer {name} in one to three sentences. Stay true to the facts (you may be vague or "
               "play them up if that is your style, but never state something false as a fact, and describe the "
               "cat's temperament and habits only as the facts do). Anything you say about cats in general must "
               "agree with the general facts above, and never hint at a risk those facts rule out (for example, "
               "telling someone with kids or dogs to be careful because the cat is FIV positive). Write only the "
               "answer.")


# Example lines in the owner's style (2026-10-11 review). Two are shown per call, drawn at random, because a fixed
# example was copied: 2,260 of 4,261 reactions in Kaggle v9 began with its "Good to know, but I'm still waiting".
REACTION_EXAMPLES = (
    "It's nice that she gets along with other cats, my old tabby could use a calm friend. I can go up to $95.",
    "Oh, kidney disease means a lot of vet visits on my budget. I'll have to come down to $60.",
    "I'm retired and on my own all day, so a cat who follows me around and sits on my lap would be wonderful "
    "company. I'll go up to $140.",
    "A cat who wants to be followed and petted needs more attention than I can give after twelve-hour shifts. "
    "I'll drop to $30.",
    "Getting along with other cats is nice, but I don't have any other cats and I don't need a clingy one, so I'll "
    "stay at $65.",
    "If she's FIV positive she should stay indoors, and I wanted a cat to take hiking with me, so I'll come down to "
    "$65.",
    "I don't have kids, so how she is with children doesn't change anything for me. I'll stay at $75, but I'd like "
    "to know how much she sheds.",
)
# Lines that give no reason (owner: 嚴禁), and the copied opener. The last four were found in the Modal run of
# 2026-10-11 (data/kaggle_out/dialogues_modal/).
VAGUE = ("sounds perfect for me", "sounds even better", "exactly what i'm looking for", "exactly what i want",
         "the kind of cat i'm looking for", "not what i'm looking for", "good to know, but i'm still waiting",
         "perfect fit", "what i was looking for", "what i care about", "fits my lifestyle")
# "She being a short-haired tabby doesn't matter": not English, and 162 of 1,413 lines in that run began this way.
BROKEN_OPENER = re.compile(r"(?:^|[.!?]\s+)(?:she|he|it|the cat|this cat)\s+being\b", re.I)
STOP = frozenset("about after again also because been before being could does from have just like more much need "
                 "only really said says should some than that their them then there these they this those very "
                 "want were what when which while will with would your she's he's it's i'm i'll i'd that's good "
                 "know great nice sounds pretty".split())
REACT_TRIES = 3


def _words(text: str) -> set[str]:
    """The first four letters of each word of four or more letters, stop words left out ("vaccines" -> "vacc"):
    two lines share a word when these sets meet."""
    return {w[:4] for w in re.findall(r"[a-z']+", text.lower()) if len(w) >= 4 and w not in STOP}


def reaction_problem(row: dict[str, Any], card: dict[str, Any], old: int, name: str, answer: str) -> str | None:
    """Why a reaction breaks a rule a program can check (owner, 2026-10-11), or None."""
    sign, says = str(row.get("sign", "")), str(row.get("says", ""))
    try:
        offer = int(row["offer"])
    except (KeyError, TypeError, ValueError):
        return "offer must be a whole number of dollars"
    problem = _offer_problem(sign, offer, old, card) or _amount_problem(says, offer) or _spoken_problem(says, name)
    if problem is None and sign == "0" and not _words(says) & _words(answer):
        problem = "an unchanged offer must name what the seller just said"
    return problem


def _offer_problem(sign: str, offer: int, old: int, card: dict[str, Any]) -> str | None:
    """The sign must agree with the move from `old` to `offer`, and the offer stay within a budget if the card has one."""
    if sign not in SIGNS:
        return 'sign must be "+", "-" or "0"'
    if (sign == "+" and offer <= old) or (sign == "-" and offer >= old) or (sign == "0" and offer != old):
        return f'sign {sign} does not match the offer going from ${old} to ${offer}'
    if "budget" in card and offer > card["budget"]:          # no budget: buyers without a limit (2026-10-12)
        return f"${offer} is above the budget of ${card['budget']}"
    return None


def _amount_problem(says: str, offer: int) -> str | None:
    """A line that says one dollar amount must say the new offer (a withdrawal, offer 0, is not checked)."""
    amounts = {int(x) for x in DOLLARS.findall(says)}
    if offer and len(amounts) == 1 and offer not in amounts:
        return f"the line says ${amounts.pop()} but the offer is ${offer}"
    return None


def _spoken_problem(says: str, name: str) -> str | None:
    """Why a buyer's spoken line breaks a rule on how it is said (dash, own name, broken opener, no reason), or None."""
    if any(dash in says for dash in DASHES):
        return "the line uses a dash"
    if re.search(rf"\b{re.escape(name)}\b", says):
        return "the buyer names themselves"
    if BROKEN_OPENER.search(says):
        return 'a sentence starts "She being ..." (or "The cat being ..."), which is not English; write a full sentence'
    vague = [v for v in VAGUE if v in says.lower().replace("\u2019", "'")]
    if vague:
        return f'"{vague[0]}" gives no reason; say what was just said and what it means for your own life'
    return None


def _reactions(m: Market, asker: str) -> Steps[None]:
    """Every buyer still bidding reacts to the last answer: sign, one or two spoken sentences that give the reason
    for the new offer, and the new offer (0 = withdraws). The reason is part of what the buyer says, in the first
    person (owner, 2026-10-11). Each buyer is asked on their own, seeing only their own card, the questions and
    answers, and their own earlier lines: when all buyers shared one prompt, reasons were taken from other buyers'
    cards (Modal run, 2026-10-11). A reaction that breaks a rule a program can check (`reaction_problem`) is asked
    again, told what was wrong, up to twice; one that still breaks it is left out and the buyer keeps their offer."""
    last = next(e for e in reversed(m.events) if e["kind"] == "answer")
    rng = random.Random(f"{facts(m.s)}|{len(m.events)}")
    examples = " and ".join(f'"{x}"' for x in rng.sample(REACTION_EXAMPLES, 2))
    shared = (f"What a knowledgeable buyer knows about cats (true; use what fits):\n{knowledge(m.s)}\n\n"
              f"Questions and answers before this one:\n{m.questions_and_answers(len(m.events) - 1)}\n\n"
              f'The seller has just answered {asker}\'s question "{last["q"]}":\n"{last["says"]}"\n\n'
              "React to this latest answer only, as the buyer described below. First decide from your own card "
              "whether it touches something you care about and whether that is good or bad for you (use real cat "
              "knowledge, e.g. what an age, a health record or a habit means): sign = \"+\" (higher offer), \"-\" "
              "(lower offer, or 0 to withdraw) or \"0\" (same offer); offer = your new offer, never above your "
              "budget. Then says = what you say out loud to the seller: one or two fluent, grammatical sentences in "
              "the first person, the way a real person talks at a market, reacting to what was just said and giving "
              f"your own reason, with the new amount if it changed. Two examples of the style: {examples} Use your "
              "own words; never copy an example or start the way it starts, and never start a sentence with \"She "
              "being\", \"He being\" or \"The cat being\". Never speak about yourself by name. Use only what the "
              "seller actually said: never react to a fact the seller has not said. Every reason agrees with the "
              "facts about cats above: never fear or hope for something those facts rule out (for example, FIV "
              "cannot pass to people or dogs). The reason names what the seller just said and how it meets or "
              "misses your own situation, in concrete words from your own life as your card describes it and "
              "nothing from anyone else's life: if your card does not mention dogs, children or another cat, you "
              "have none. Saying only that the cat is or is not what you want (\"a perfect fit\", \"exactly what I "
              "was looking for\") is not a reason and is not allowed. Facts from earlier answers are already in your "
              "offer, so they are never the reason again, even when the seller repeats them. When this answer "
              "touches nothing you care about, keep the offer and say what was just said and why it does not matter "
              "for your own situation; you may add what you would still like to know, which must be something no "
              "answer has covered yet. When it does touch your card but you keep the offer, say why. Withdrawing is "
              "final: withdraw only when something the seller actually said makes this cat impossible for you (for "
              "example a breeder hearing she cannot be bred), not merely less attractive or not yet known (that is "
              "a lower or unchanged offer); then sign \"-\", offer 0, and name what was said and why it rules the "
              "cat out for you.\n\n")
    todo, notes = list(m.present()), {}
    for _ in range(REACT_TRIES):
        prompts = [shared + f"{m.card_line(n)}\nWhat you have said so far:\n{m.own_lines(n)}{notes.get(n, '')}\n\n"
                   'Return {"sign": ..., "says": ..., "offer": <dollars>}.' for n in todo]
        rows = yield from _json_each(prompts)
        notes, still = {}, []
        for name, row in zip(todo, rows, strict=True):
            problem = (reaction_problem(row, m.cards[name], m.offers[name], name, last["says"])
                       if isinstance(row, dict) else "no reaction was given")
            if problem is None:
                m.events.append({"kind": "react", "name": name, "old": m.offers[name], "offer": int(row["offer"]),
                                 "sign": str(row["sign"]), "says": str(row["says"])})
                m.offers[name] = int(row["offer"])
            else:
                still.append(name)
                notes[name] = f"\n\nYour last reaction broke a rule ({problem}); write it again."
        todo = still
        if not todo:
            return
        m.react_retried += len(todo)
    m.react_retried -= len(todo)
    m.react_dropped += len(todo)


def _round(m: Market, k: int, asked: set[str]) -> Steps[bool]:
    """Round k: one buyer who has not asked asks, the seller answers them, everyone reacts. False if no one asked."""
    left = [n for n in m.present() if n not in asked]
    q = yield from _json(f"Conversation so far:\n{m.transcript()}\n\nBuyers and their cards:\n{m.card_text()}\n\n"
              f"What a knowledgeable buyer knows about cats (true; use what fits):\n{knowledge(m.s)}\n\n"
              f"Choose ONE buyer who has not asked a question yet ({', '.join(left)}) and write the question that "
              "buyer most needs answered, given their card, what they know about cats and what has already been "
              'said. Do not repeat a question already asked. Return {"name": ..., "question": ...}.')
    if not isinstance(q, dict) or q.get("name") not in set(left):
        return False
    name = q["name"]
    asked.add(name)
    m.events.append({"kind": "ask", "round": k, "name": name, "offer": m.offers[name], "says": q["question"]})
    m.events.append({"kind": "answer", "round": k, "to": name, "q": q["question"],
                     "says": (yield from _answer(m, name, q["question"]))})
    yield from _reactions(m, name)
    return True


MAX_THINGS = 3                       # exactly this many things in every scene (owner, 2026-10-11)


# The owner's examples (2026-10-11), two shown per sale at random: a fixed pair was copied (in Kaggle v9, 312 of 392
# scenes used two or more of its things; bed and couch 241 times each).
IMAGINE_EXAMPLES = (
    ("works long hours", "couch, alarm clock and bed", "With my long hours she'll need a good couch to nap on, an "
     "alarm clock so I never get so busy that I forget to feed her, and my bed to curl up on when I finally get home."),
    ("lost an old cat", "picture frame, couch and pillow", "I'd show her my old cat's photo in this frame, and let her "
     "have the couch he loved most and his old pillow."),
    ("cares how things look", "flower, vase and pillow", "I'll buy fresh flowers every day and keep them in a vase "
     "next to her to set off how pretty she is, and give her a lovely pillow of her own."),
    ("hikes every weekend", "backpack, map and mountain", "I want to take her up the mountain with me, riding in my "
     "backpack right next to the map."),
    ("lives in a small apartment", "skyscraper, house plant and bed", "From my little apartment she can watch the "
     "skyscraper across the street, nap by my house plant and share the one bed I have room for."),
)
IMAGINE_TRIES = 4                    # the first answer and up to three rewrites (with 3, 77 of 142 scenes were dropped)
DRAWABLE_THINGS = frozenset(DRAWABLE) - {"cat"}


# Class names that are also everyday words ("keep an eye on", "the key thing", "it matches", "not a fan", "my house"):
# a line using them is not taken to mention that object. Body parts and shapes are left out for the same reason.
EVERYDAY_WORDS = frozenset({"eye", "key", "matches", "fan", "line", "face", "house", "hand", "foot", "leg", "arm", "ear",
                            "nose", "mouth", "tooth", "finger", "toe", "knee", "beard", "moustache", "smiley face",
                            "circle", "square", "triangle", "hexagon", "octagon", "zigzag", "squiggle"})
# Things buyers name that the drawing GPT cannot draw (none of these is one of its classes; tested). Not complete:
# anything new found while spot-checking is added here.
NOT_DRAWABLE = ("window", "blanket", "rug", "carpet", "sofa", "shelf", "bookshelf", "bookcase", "bowl", "toy",
                "litter box", "litter", "scratching post", "cat tree", "catio", "cushion", "curtain", "desk", "lamp",
                "plant", "yard", "backyard", "balcony", "food", "treat", "collar", "leash", "carrier", "box", "mat",
                "boots", "trail", "porch")


def _mentions(names: frozenset[str] | tuple[str, ...], line: str) -> list[str]:
    """Names (singular or with an s) that appear in a line as whole words."""
    low = line.lower()
    return [n for n in names if re.search(rf"\b{re.escape(n)}s?\b", low)]


def _named(thing: str, line: str) -> bool:
    """Whether a line names a thing (by its last word: "alarm clock" by "clock", "picture frame" by "frame")."""
    return re.search(rf"\b{re.escape(thing.split()[-1].lower())}", line.lower()) is not None


def imagine_problem(row: dict[str, Any]) -> str | None:
    """Why an imagined scene breaks the owner's rules (2026-10-11), or None: every thing must be on the drawing GPT's
    list and named in the line, and the line names no other object: no other drawable class (everyday words aside)
    and nothing on NOT_DRAWABLE."""
    things = row.get("things")
    if not isinstance(things, list) or len(things) != MAX_THINGS:
        return f"things must be a list of exactly {MAX_THINGS} items"
    off = [str(t) for t in things if str(t) not in DRAWABLE_THINGS]
    if off:
        return f"{', '.join(off)} {'is' if len(off) == 1 else 'are'} not on the list of things that can be drawn"
    says = str(row.get("says", ""))
    unsaid = [str(t) for t in things if not _named(str(t), says)]
    if unsaid:
        return f"the line never names {', '.join(unsaid)} or says what it is for"
    picked = [str(t) for t in things]
    unpicked = sorted(n for n in _mentions(DRAWABLE_THINGS - EVERYDAY_WORDS, says)      # "clock" inside "alarm clock"
                      if not any(re.search(rf"\b{re.escape(n)}\b", t) for t in picked))
    if unpicked:
        return f"the line mentions {', '.join(unpicked)} without picking it; name only the things picked"
    other = [w for w in _mentions(NOT_DRAWABLE, says) if not any(w in t for t in things)]
    if other:
        return f"the line mentions {', '.join(other)}, which cannot be drawn; mention only the things picked"
    return None


def _scene_problem(row: object, taken: list[set[str]]) -> str | None:
    """Why a scene cannot be kept: it breaks a rule (`imagine_problem`) or shares two things with a scene already kept."""
    if not isinstance(row, dict):
        return "no answer was given"
    problem = imagine_problem(row)
    if problem is None:
        same = next((t for t in taken if len(t & {str(x) for x in row["things"]}) >= 2), None)
        if same:
            problem = (f"another buyer already pictured {', '.join(sorted(same & set(map(str, row['things']))))}; "
                       "pick things from your own life")
    return problem


def _imagine(m: Market) -> Steps[None]:
    """Every buyer still bidding pictures the cat at their own place: three drawable things around it, chosen from
    their own card and what was said, and one or two spoken sentences naming each thing and what it is for. This
    replaces the rule-based scenes (scenes.py) as training data: the buyer model learns what to picture (owner,
    2026-10-11). Each buyer is asked on their own (their card, the answers, their own lines) and is told the words
    that cannot be drawn. A scene that breaks a rule (`imagine_problem`), or shares two things with a scene already
    kept, is asked again, told what was wrong, up to three times; one that still breaks it is dropped, never kept."""
    things = ", ".join(sorted(DRAWABLE_THINGS))
    rng = random.Random(f"{facts(m.s)}|imagine")
    examples = " and ".join(f'a buyer who {who} picks {picked}: "{line}"'
                            for who, picked, line in rng.sample(IMAGINE_EXAMPLES, 2))
    shared = (f"Things that can be drawn: {things}.\n\nThe questions and answers in this sale:\n"
              f"{m.questions_and_answers()}\n\nThe questions are over. Picture this cat at your own place, as your "
              "card, your life and what you heard suggest. things = exactly three things that would be around the cat "
              "in that picture, each written exactly as it appears in the list above (not the cat itself; nothing "
              "that is not on the list). Pick things from your own life (your work, your hobbies, your family, your "
              "home) as your card describes it, not furniture anyone has. says = one or two sentences in the first "
              "person, the way a real person talks, that name every thing you picked and what it is for in your life "
              "with this cat. Mention no other object at all: no other thing from the list, and none of these, which "
              f"cannot be drawn: {', '.join(NOT_DRAWABLE)}. Two examples: {examples}. Use your own things and words; "
              "do not reuse the examples or their things unless they truly belong to your life.\n\n")
    todo, notes, taken = list(m.present()), {}, []
    for _ in range(IMAGINE_TRIES):
        prompts = [shared + f"{m.card_line(n)}\nWhat you said during the sale:\n{m.own_lines(n)}{notes.get(n, '')}"
                   '\n\nReturn {"things": [...], "says": ...}.' for n in todo]
        rows = yield from _json_each(prompts)
        notes, still = {}, []
        for name, row in zip(todo, rows, strict=True):
            problem = _scene_problem(row, taken)
            if problem is None:
                taken.append({str(x) for x in row["things"]})
                m.events.append({"kind": "imagine", "name": name, "things": [str(t) for t in row["things"]],
                                 "says": str(row.get("says", ""))})
            else:
                still.append(name)
                notes[name] = f"\n\nYour last answer broke a rule ({problem}); write it again."
        todo = still
        if not todo:
            return
        m.imagine_retried += len(todo)
    m.imagine_retried -= len(todo)
    m.imagine_dropped += len(todo)


def sale_steps(s: Sale, rounds: int = N_BUYERS) -> Steps[Market]:
    """One sale as a generator: it yields each prompt for the teacher and is sent back the reply."""
    m = Market(s)
    yield from _opening(m)
    asked: set[str] = set()
    for k in range(1, rounds + 1):
        if not (yield from _round(m, k, asked)):
            break
    if rounds == N_BUYERS and m.present():
        yield from _imagine(m)
        best = max(m.present(), key=m.offers.get)
        m.events.append({"kind": "sold", "name": best, "price": m.offers[best]})
    return m


def _render_react(e: Event) -> str:
    """A reaction with its sign, what was said and the offer change ("out" for a withdrawal)."""
    new = f"${e['offer']}" if e["offer"] else "out"
    why = f" | why: {e['why']}" if e.get("why") else ""          # only sales made before 2026-10-11 have a WHY
    return f"<|react|> {e['name']} ({e['sign']}){why} | says: {e['says']} | ${e['old']}->{new}"


RENDER: dict[str, Callable[[Event], list[str]]] = {
    "open": lambda e: [f"<|open|> {e['name']} ${e['offer']} | {e['says']}"],
    "ask": lambda e: [f"<|round {e['round']}|>", f"<|ask|> {e['name']} ${e['offer']} | {e['says']}"],
    "answer": lambda e: [f"<|answer to {e['to']}|> {e['says']}"],
    "react": lambda e: [_render_react(e)],
    "sold": lambda e: [f"<|sold|> {e['name']} ${e['price']}"],
    "imagine": lambda e: [f"<|imagine|> {e['name']} | things: {', '.join(e['things'])} | says: {e['says']}"],
}


def render(m: Market) -> str:
    """The training layout: cards, opening, then each round between markers, every reaction with its sign, what was
    said and the offer change (sales made before 2026-10-11 also carry a separate WHY)."""
    out = ["<|sale|> cat"]
    out += [f"<|buyer|> {n} | {c['persona']} | cares about: {', '.join(c['cares_about'])}"
            + (f" | budget ${c['budget']}" if "budget" in c else "")
            for n, c in m.cards.items()]
    for e in m.events:
        if e["kind"] not in RENDER:
            raise ValueError(f"unknown event kind {e['kind']!r}")
        out += RENDER[e["kind"]](e)
    return "\n".join(out)


DOLLARS = re.compile(r"\$\s?(\d+)")


def check(m: Market) -> dict[str, int]:
    """Rule counts over the reactions: offer moving against its own sign; "+" or "-" with the offer unchanged; above
    the buyer's budget; a dollar amount said that is not the new offer (when only one amount is said); a buyer
    naming themselves; over every spoken line, dashes; imagined things that the drawing GPT cannot draw, and imagined
    things the buyer's line never names (matched on the last word: "alarm clock" by "clock")."""
    reacts = [e for e in m.events if e["kind"] == "react"]
    against = sum(1 for e in reacts if SIGNS.get(e["sign"], 0) * (e["offer"] - e["old"]) < 0
                  or (e["sign"] == "0" and e["offer"] != e["old"]))
    still = sum(1 for e in reacts if e["sign"] in ("+", "-") and e["offer"] == e["old"])
    over = sum(1 for e in reacts if e["offer"] > m.cards[e["name"]].get("budget", float("inf")))   # none: no limit
    said = [[int(x) for x in DOLLARS.findall(e["says"])] for e in reacts]
    wrong_amount = sum(1 for e, amounts in zip(reacts, said, strict=True)
                       if len(set(amounts)) == 1 and e["offer"] and amounts[0] != e["offer"])
    own_name = sum(1 for e in reacts if re.search(rf"\b{re.escape(e['name'])}\b", e["says"]))
    dashes = sum(1 for e in m.events if any(d in e.get("says", "") for d in DASHES))
    pictured = [t for e in m.events if e["kind"] == "imagine" for t in e["things"]]
    unsaid = sum(1 for e in m.events if e["kind"] == "imagine" for t in e["things"] if not _named(t, e["says"]))
    return {"imagined": sum(e["kind"] == "imagine" for e in m.events), "things_pictured": len(pictured),
            "things_not_drawable": sum(t not in DRAWABLE_THINGS for t in pictured), "things_not_said": unsaid,
            "imagine_retried": getattr(m, "imagine_retried", 0), "imagine_dropped": getattr(m, "imagine_dropped", 0),
            "react_retried": getattr(m, "react_retried", 0), "react_dropped": getattr(m, "react_dropped", 0),
            "reactions": len(reacts), "offer_against_sign": against, "sign_without_move": still,
            "over_budget": over, "amount_said_differs": wrong_amount, "names_self": own_name,
            "lines_with_dash": dashes}


def _items(m: Market, start: int, end: int) -> list[str]:
    """The answer at `start`, unnumbered, and the reactions after it, up to `end`, numbered for the reviewer."""
    answer = m.events[start]
    out = [f"SELLER to {answer['to']} (who asked: {answer['q']}): {answer['says']}"]
    for i in range(start + 1, end):
        e = m.events[i]
        card = m.cards[e["name"]]
        out.append(f"[{i}] {e['name']} (who {card['persona']}; cares about {', '.join(card['cares_about'])}) "
                   f"says: {e['says']} | sign {e['sign']} | ${e['old']} -> ${e['offer']}")
    return out


def judge_steps(m: Market) -> Steps[list[dict[str, Any]]]:
    """The teacher as a reviewer of buyers, one round at a time: each reaction against only what had been said by
    then, the facts about cats (all topics) and the buyer's own card. Sellers may exaggerate or lie, so their answers
    are not judged and the cat's true facts are kept from the reviewer. Returns the reactions it finds wrong or
    doubtful."""
    every_fact = "\n".join(f"- {f}" for t in TOPICS.values() for f in t["facts"])
    problems: list[dict[str, Any]] = []
    for start in (i for i, e in enumerate(m.events) if e["kind"] == "answer"):
        end = next((i for i in range(start + 1, len(m.events)) if m.events[i]["kind"] != "react"), len(m.events))
        rows = yield from _json(
            f"You are checking buyers' reactions in training data for mistakes of logic.\nTrue facts about cats:\n"
            f"{every_fact}\n\nConversation before this round:\n{m.transcript(start)}\n\nThis round:\n"
            + "\n".join(_items(m, start, end)) + "\n\n"
            "The seller may exaggerate or lie about this cat and is not being checked; a buyer who takes the seller at their word "
            "is reasoning correctly. A reaction may use only what has been said so far, this round's answer included, "
            "and never something the seller has not said. It is wrong if the reason it gives misreads what was said, "
            "contradicts the facts about cats, points the wrong way for that buyer's card, or withdraws ($0) over a "
            "drawback that does not rule the cat out for that buyer. Sign 0 with no reason is right when the answer "
            "touches nothing on the buyer's card. Flag a reaction only if no sensible buyer with that card could decide that "
            "way; another reasonable decision existing is not a mistake. Return a JSON list of only the numbered "
            'reactions that are wrong or doubtful (an empty list if every one is fine): {"id": <number>, '
            '"verdict": "wrong" | "doubtful", "problem": "<short>"}.')
        if not isinstance(rows, list):
            rows = [{"id": start, "verdict": "doubtful", "problem": "the reviewer gave no list"}]
        problems += [r for r in rows if isinstance(r, dict) and r.get("verdict") in ("wrong", "doubtful")]
    return problems


def knowledge_steps(topic: str, form: str) -> Steps[dict[str, Any]]:
    """One knowledge text written by the teacher, then checked by it against the topic's facts."""
    text = yield knowledge_prompt(topic, form)
    facts_list = "\n".join(f"- {f}" for f in TOPICS[topic]["facts"])
    rows = yield from _json(f"You are checking training data for mistakes.\nTrue facts:\n{facts_list}\n\nText:\n{text}\n\n"
                            "List every sentence of the text that contradicts the facts, adds a medical claim or number "
                            "that is not in them, or contradicts itself. Return a JSON list (empty if there are none): "
                            '{"sentence": "...", "problem": "<short>"}.')
    problems = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else \
        [{"sentence": None, "problem": "the reviewer gave no list"}]
    return {"topic": topic, "form": form, "text": text, "problems": problems}


def drive(steps: Steps[T], ask: Ask) -> T:
    """Run a generator of prompts to the end, answering each prompt (or each prompt of a list) with ask."""
    try:
        prompt = next(steps)
        while True:
            prompt = steps.send([ask(p) for p in prompt] if isinstance(prompt, list) else ask(prompt))
    except StopIteration as done:
        return done.value


def run_many(all_steps: list[Steps[T]], ask_many: Callable[[list[str]], list[str]]) -> list[T]:
    """Run many generators side by side: each turn, every waiting prompt (a list of prompts counts as its prompts)
    goes to ask_many in one batch (how a GPU server such as vLLM is kept busy), and each reply is sent back to the
    generator that asked."""
    results: list[Any] = [None] * len(all_steps)
    waiting: dict[int, Any] = {}
    for i, steps in enumerate(all_steps):
        try:
            waiting[i] = next(steps)
        except StopIteration as done:
            results[i] = done.value
    while waiting:
        ids = list(waiting)
        batch = [p for i in ids for p in (waiting[i] if isinstance(waiting[i], list) else [waiting[i]])]
        replies = iter(ask_many(batch))
        for i in ids:
            if isinstance(waiting[i], list):
                reply: Any = [next(replies) for _ in waiting[i]]
            else:
                reply = next(replies)
            try:
                waiting[i] = all_steps[i].send(reply)
            except StopIteration as done:
                results[i] = done.value
                del waiting[i]
    return results


def sale(ask: Ask, s: Sale, rounds: int = N_BUYERS) -> Market:
    """One sale written by `ask` from start to end (see sale_steps)."""
    return drive(sale_steps(s, rounds), ask)


def judge(ask: Ask, m: Market) -> list[dict[str, Any]]:
    """The teacher as a reviewer (see judge_steps)."""
    return drive(judge_steps(m), ask)


def sale_and_review(s: Sale, rounds: int = N_BUYERS) -> Steps[tuple[Market, list[dict[str, Any]]]]:
    """A sale, then the teacher's review of its reactions, as one generator (for run_many)."""
    m = yield from sale_steps(s, rounds)
    return m, (yield from judge_steps(m))


def mlx_teacher(path: Path = LOCAL_TEACHER, max_tokens: int = 2500) -> Ask:
    """The teacher on this Mac's GPU through MLX: temperature 0.8, top-p 0.95, as on the cloud."""
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(str(path))
    sampler = make_sampler(temp=0.8, top_p=0.95)

    def ask(prompt: str) -> str:
        msgs = [{"role": "system", "content": SETTING}, {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return generate(model, tok, text, max_tokens=max_tokens, sampler=sampler).strip()
    return ask


def main() -> None:
    """Command line: one sale on this Mac, printed and saved under runs/buyer_dialogues/."""
    parser = argparse.ArgumentParser(description="One cat sale written by the teacher model on this Mac.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=N_BUYERS)
    args = parser.parse_args()
    s = seed_sale(random.Random(args.seed))
    t0 = time.time()
    m = sale(mlx_teacher(), s, args.rounds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"cat_seed{args.seed}_rounds{args.rounds}_mac.json"
    report = {"seed": s, "events": m.events, "cards": m.cards, "check": check(m), "seconds": round(time.time() - t0)}
    path.write_text(json.dumps(report, indent=1))
    print(f"cat: {facts(s)}\nseller: {s['seller_style']}\n\n{render(m)}\n\ncheck: {report['check']}, "
          f"{report['seconds']} s\nsaved {path}")


if __name__ == "__main__":
    main()
