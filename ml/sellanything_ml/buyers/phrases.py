"""v1 phrase pools: how buyers sound before the model exists (docs/plan.md D8).

These lines are also the raw material for training data v1, so they're written to vary in
wording while keeping the meaning tied to the label (Δ) they illustrate.
"""

from __future__ import annotations

import random
from collections.abc import Collection

# Wrappers around the schema's canonical question.
ASK = [
    "{q}",
    "Quick question: {lq}",
    "Before I decide, {lq}",
    "Mind if I ask? {q}",
    "So, {lq}",
    "Can you tell me, {lq}",
    "I'm curious. {q}",
    "One thing I need to know: {lq}",
]

# Reactions by interest change. {noun} is the item noun.
REACT = {
    2: [
        "Oh, that's exactly what I was hoping to hear.",
        "Perfect, that's a big plus for me.",
        "Now we're talking. I really like that.",
        "That's great news, I'm a lot more interested now.",
        "Honestly, that makes this {noun} perfect for me.",
    ],
    1: [
        "Nice, that's good to know.",
        "Okay, I like that.",
        "Good, that helps.",
        "That's a point in its favor.",
        "Alright, that works for me.",
    ],
    0: [
        "Okay, noted.",
        "Fair enough.",
        "Hm, okay.",
        "Got it.",
        "Doesn't matter much to me either way.",
    ],
    -1: [
        "Hm, that's not ideal.",
        "Oh, that's a bit of a letdown.",
        "I was hoping for something better there.",
        "That makes me a little less sure.",
        "Okay... that's a minus for me.",
    ],
    -2: [
        "Ugh, that's a real problem for me.",
        "Oh no, that's a dealbreaker-level issue.",
        "That really doesn't work for me.",
        "Yikes. That changes things a lot.",
        "Honestly, that's the opposite of what I need.",
    ],
}

# Δ = 0 but the answer moved the objective value a lot (factor ≥ 1.2 or ≤ 0.8): everyone's price
# changes, so the buyer comments on value rather than on their own taste.
VALUE_UP = [
    "Oh, that makes it worth quite a bit more.",
    "Well, that definitely adds to the value.",
    "That's a big plus for the price, I'll admit.",
]
VALUE_DOWN = [
    "Hm, that brings the value down a fair bit.",
    "That's going to knock the price down.",
    "Okay, that lowers what it's worth.",
]
VALUE_FACTOR = 0.2   # |factor − 1| at or above this counts as "a lot"

LEAVE = [
    "Sorry, I don't think this {noun} is for me. Good luck!",
    "I'm going to pass. Thanks anyway.",
    "Yeah, this isn't going to work for me. I'm out.",
]

RAISE = [
    "I'll go ${amount}.",
    "Okay, ${amount} from me.",
    "I can beat that: ${amount}.",
    "Fine, ${amount}. Final-ish offer.",
]

REFUSE = [
    "{A_noun}? Sorry, that's not something I'd buy.",
    "I'm not buying {a_noun}. That's not even something you can sell.",
    "No thanks, I'll pass on that.",
    "Ha, {a_noun}? I don't think so.",
    "{A_noun}? You can't be serious. No deal.",
]


def _lower_first(text: str) -> str:
    """Lower-case the first letter unless the first word is an acronym (LED, AA, USB)."""
    first = text.split(" ", 1)[0].strip(",.?!")
    if len(first) > 1 and first.isupper():
        return text
    return text[:1].lower() + text[1:]


# Refuse items that read wrong with "a/an": landmarks and mass nouns.
MASS_NOUNS = {"rain", "grass", "lightning", "camouflage", "yoga", "animal migration", "stitches",
              "stairs", "squiggle", "zigzag"}
ONE_OF_A_KIND = {"sun", "moon"}       # "the sun", not "a sun"


def with_article(noun: str) -> str:
    """'a tiger', 'an elbow', 'The Eiffel Tower', 'rain'."""
    if noun.startswith("The ") or noun in MASS_NOUNS:
        return noun
    if noun in ONE_OF_A_KIND:
        return "the " + noun
    return ("an " if noun[0].lower() in "aeiou" else "a ") + noun


def _cap(text: str) -> str:
    """Upper-case only the first letter (str.capitalize would lower the rest)."""
    return text[:1].upper() + text[1:]


def ask(question: str, rng: random.Random) -> str:
    """The schema question in one of the ASK wordings."""
    return rng.choice(ASK).format(q=question, lq=_lower_first(question))


def react(delta: int, noun: str, rng: random.Random, factor: float = 1.0) -> str:
    """Reaction line for an interest change; with Δ = 0, a big objective change gets a value comment."""
    if delta == 0 and factor >= 1 + VALUE_FACTOR - 1e-9:
        return rng.choice(VALUE_UP)
    if delta == 0 and factor <= 1 - VALUE_FACTOR + 1e-9:
        return rng.choice(VALUE_DOWN)
    return rng.choice(REACT[delta]).format(noun=noun)


def leave(noun: str, rng: random.Random) -> str:
    """What a buyer says when their interest hits the walk-away threshold."""
    return rng.choice(LEAVE).format(noun=noun)


def raise_bid(amount: float, rng: random.Random) -> str:
    """A buyer outbidding the leader, with the amount written without trailing zeros ("$1,250", "$7.5")."""
    shown = f"{amount:,.2f}".rstrip("0").rstrip(".")
    return rng.choice(RAISE).format(amount=shown)


def refuse_lines(noun: str, n: int, rng: random.Random) -> list[str]:
    """n refusals for one round, all different while the pool lasts."""
    pool = rng.sample(REFUSE, len(REFUSE))
    a_noun = with_article(noun)
    return [pool[i % len(pool)].format(a_noun=a_noun, A_noun=_cap(a_noun)) for i in range(n)]


# Turning down a 'risky' item, in terms of the trait that makes the buyer refuse (engine.refuses_risky).
RISKY_REFUSE = {
    "parent": ["Not with kids in the house. No way.", "I have little kids, that's way too dangerous.",
               "Absolutely not, I've got a family to think about."],
    "retiree": ["At my age? That's far too much trouble.", "I like a quiet life. That's not for me.",
                "No thank you, that's too dangerous for an old-timer like me."],
    "apartment_dweller": ["That's not going in my apartment.", "My landlord would kill me. Pass.",
                          "In a small apartment? No chance."],
}


def risky_refusal(trait: str, rng: random.Random) -> str:
    """Why a buyer with this trait walks away from a risky item at once."""
    return rng.choice(RISKY_REFUSE[trait])


# --- persona-grounded reasons (v1 data generation) -------------------------------------------
# Tied to tags, so a buyer explains *why* in terms of who they are. The reason chosen for a
# reaction is the tag that contributed most to Δ, so the text always agrees with the label.

ASK_REASON = {
    "condition": ["I'm pretty picky about condition.", "I want something in great shape.",
                  "Condition matters a lot to me."],
    "space": ["My place is pretty small.", "I don't have much room.", "Space is tight where I live."],
    "low_maintenance": ["I don't have a lot of time for upkeep.", "I need something easy to look after.",
                        "I'm hardly ever home."],
    "family": ["It's for my kids.", "I've got little ones at home.", "It needs to be family friendly."],
    "status": ["I like to have the good stuff.", "I collect nice things.", "Brand matters to me, honestly."],
    "performance": ["I need it to actually work well.", "Performance is what I care about.",
                    "It has to do the job."],
    "style": ["I care how it looks.", "Looks matter to me.", "It has to look good in my place."],
    "active": ["I'm outdoors most weekends.", "I'd be taking it on trips.", "I'm pretty active."],
}

REASON_P = 0.7     # share of moved offers whose reaction says why
REACT_REASON = {
    ("condition", 1): ["I like that it's been looked after.", "Good condition is a big deal for me."],
    ("condition", -1): ["I really wanted something in better shape.", "That's more wear than I'd like."],
    ("space", 1): ["That would fit my place perfectly.", "Great, I don't have much room."],
    ("space", -1): ["That's too big for where I live.", "I just don't have the room for that."],
    ("low_maintenance", 1): ["Less work for me, love it.", "Perfect, I don't have time for upkeep."],
    ("low_maintenance", -1): ["That sounds like a lot of work.", "I don't have time to deal with that."],
    ("family", 1): ["That's great for the kids.", "That makes it a good fit for my family."],
    ("family", -1): ["I'm not sure that's safe with kids around.", "That's not great for my family."],
    ("status", 1): ["Now that's something special.", "I appreciate a quality name."],
    ("status", -1): ["I was hoping for something a bit more special.", "That's a bit basic for me."],
    ("performance", 1): ["That means it'll actually do the job well.", "Sounds like it performs great."],
    ("performance", -1): ["That doesn't sound like it'll work very well.", "I need it to perform better."],
    ("style", 1): ["That sounds really nice looking.", "I love the look of that."],
    ("style", -1): ["That's not really my style.", "Hm, I don't love the look of that."],
    ("active", 1): ["Perfect for my weekends outdoors.", "That's great for an active lifestyle."],
    ("active", -1): ["Not great for how active I am.", "That won't keep up with me."],
}

# The condition pools above speak of things (wear, shape, looked after). When the answer is about a
# living thing's health, food's freshness or an animal's age, the same tag and sign need other words:
# "That's more wear than I'd like." about a cat's shots read wrong in play (design-v2 §6.1, §6.6).
# Chosen by the attribute being answered, not by the item: a cat's litter training also carries a
# condition signal and is not about health. Each topic pool has as many lines as the condition pool it
# replaces, so generated data differs only in these lines (the random draws stay the same).
HEALTH_ATTRS = {"health", "queen", "colony"}
FRESH_ATTRS = {"freshness", "fresh", "ripeness", "sprouted", "color", "snap", "state", "frozen"}

TOPIC_ASK_REASON = {
    "health": ["I want a healthy animal.", "Health is the first thing I check.", "I can't take on big vet bills."],
    "plant_health": ["I want a healthy plant.", "Health is the first thing I check.",
                     "I don't want one that's struggling."],
    "fresh": ["I want it fresh.", "Freshness matters a lot to me.", "I don't want anything going off."],
    "animal_age": ["I want one with lots of good years ahead.", "Age matters to me with animals.",
                   "I'd rather not take on an old one."],
}
TOPIC_REACT_REASON = {
    ("health", 1): ["Good to hear it's healthy.", "A healthy animal is what I want."],
    ("health", -1): ["I'd worry about its health.", "That sounds like vet bills to me."],
    ("plant_health", 1): ["Good, a healthy plant is what I want.", "I like that it's thriving."],
    ("plant_health", -1): ["That sounds like it's struggling.", "I'd worry it won't make it."],
    ("fresh", 1): ["I like that it's fresh.", "Fresh is exactly what I'm after."],
    ("fresh", -1): ["That's not as fresh as I'd like.", "I'd worry it's going off."],
    ("animal_age", 1): ["Good, plenty of years ahead.", "A young one is just what I wanted."],
    ("animal_age", -1): ["I'd rather have a younger one.", "An older animal worries me a bit."],
}


def topic(attr_name: str, group: str, item_attrs: Collection[str]) -> str | None:
    """What a condition answer is about, when the generic wording would not fit. An animal is a
    pets_livestock item or a risky one with a temperament (the wild animals; risky objects have none)."""
    animal = group == "pets_livestock" or (group == "risky" and "temperament" in item_attrs)
    if attr_name in HEALTH_ATTRS:
        return "plant_health" if group == "plants" else "health"
    if attr_name in FRESH_ATTRS:
        return "fresh"
    if attr_name == "age" and animal:
        return "animal_age"
    return None


ANSWER = [
    "{a}",
    "{a}",
    "Yeah, {la}",
    "Honestly? {a}",
    "{a} I promise.",
    "Let me think... {la}",
    "Good question. {a}",
    "So, {la}",
]


def answer(text: str, rng: random.Random) -> str:
    """The seller's answer from the schema, sometimes with a few words in front ("Honestly? ...")."""
    return rng.choice(ANSWER).format(a=text, la=_lower_first(text))


ASK_AGAIN = [
    "Sorry, I didn't catch that.",
    "Hm, which one do you mean?",
    "Sorry, I'm not sure what you mean.",
    "Wait, so which is it?",
]


def ask_again(question: str, rng: random.Random) -> str:
    """When a typed answer matched none of the options: the asker repeats the question."""
    return f"{rng.choice(ASK_AGAIN)} {question}"


def ask_with_reason(question: str, reason_tag: str | None, rng: random.Random,
                    about: str | None = None) -> str:
    """`about`: topic() of the attribute asked about; it changes only the wording of a condition reason."""
    line = ask(question, rng)
    if reason_tag and rng.random() < 0.6:
        pool = TOPIC_ASK_REASON[about] if about and reason_tag == "condition" else ASK_REASON[reason_tag]
        line = f"{line} {rng.choice(pool)}"
    return line


def react_with_reason(delta: int, noun: str, factor: float, reason_tag: str | None,
                      rng: random.Random, about: str | None = None) -> str:
    """A reaction to an answer; when the offer moved because of `reason_tag`, usually (REASON_P) followed by a
    reason from that tag's pool, or from the topic's pool (`about`, e.g. "health") for a condition answer."""
    line = react(delta, noun, rng, factor)
    if delta != 0 and reason_tag and rng.random() < REASON_P:
        sign = 1 if delta > 0 else -1
        pool = TOPIC_REACT_REASON[(about, sign)] if about and reason_tag == "condition" \
            else REACT_REASON[(reason_tag, sign)]
        line = f"{line} {rng.choice(pool)}"
    return line


# --- checking reasons in model lines --------------------------------------------------------------
# The buyer model learned its reasons from these pools, so a reason in a model line can be traced
# back to the tag it stands for, and checked against the answer and the buyer (voice.py).
REACT_REASON_OF = {p: key for key, pool in REACT_REASON.items() for p in pool} | \
    {p: ("condition", sign) for (_, sign), pool in TOPIC_REACT_REASON.items() for p in pool}
ASK_REASON_OF = {p: tag for tag, pool in ASK_REASON.items() for p in pool} | \
    {p: "condition" for pool in TOPIC_ASK_REASON.values() for p in pool}


def react_reasons_in(line: str) -> set[tuple[str, int]]:
    """(tag, +1/-1) of every reaction reason from the pools that appears in the line."""
    return {key for phrase, key in REACT_REASON_OF.items() if phrase in line}


def ask_reasons_in(line: str) -> set[str]:
    """Tags of every question reason from the pools that appears in the line."""
    return {tag for phrase, tag in ASK_REASON_OF.items() if phrase in line}
