"""Typed answers: map what the player wrote ("shes like 12") to one of the question's options.

The buyers only understand options (each one moves offers by its factor and signals), so typed text
has to land on exactly one of them, or on none: then the buyer asks again. Three rules, tried in order
and each able to narrow the choice for the next:

1. Yes or no. If the text starts with "yes"/"yeah"/"nope"/"not"..., keep the options that say the same
   ("Yes, never misses." / "Not reliably yet."; an option with a "doesn't" or "won't" in it counts as
   a no when nothing starts with one).
2. Numbers. If two or more options carry a quantity of the same kind (an age, kilos, people, a decade),
   the typed quantity picks the nearest one ("12" -> "She's eleven." when asked how old).
3. Words. Shared words, each weighted by how few options use it, so "used" picks "Used, but in good
   shape." over "Like new, barely used." A tie means we can't tell.

No model: it is rules on the option texts, so every pick can be explained. Measured on hand-written
answers in tests/fixtures/typed_answers_{dev,test}.tsv (docs/buyer-ai.md).
"""

from __future__ import annotations

import datetime
import math
import re
from dataclasses import dataclass

from .schema import Attribute, Option

MAX_CHARS = 120
THIS_YEAR = datetime.date.today().year

YES = {"yes", "yeah", "yep", "yup", "ya", "yea", "sure", "definitely", "absolutely", "certainly", "totally",
       "correct", "y", "ok", "okay"}
NO = {"no", "nope", "nah", "not", "never", "none", "n", "negative"}
NEGATIONS = NO | {"doesnt", "dont", "wont", "cant", "isnt", "arent", "wasnt", "havent", "hasnt", "didnt", "nothing"}

STOPWORDS = {"a", "an", "the", "it", "its", "is", "are", "was", "be", "been", "i", "im", "ive", "my", "me", "she",
             "shes", "he", "hes", "her", "his", "him", "they", "theyre", "them", "this", "that", "thats", "of",
             "to", "in", "for", "and", "or", "with", "very", "really", "just", "pretty", "so", "at", "from",
             "but", "um", "uh", "well", "kind", "sort", "one", "ones", "has", "have", "got", "too", "quite"}

# Words a player might use for the words in the options. Each set is one meaning; a typed word matches
# an option word when both are in the same set. Kept small and general (not per item).
SYNONYMS = [
    {"torn", "ripped", "rip", "tear", "torn"},
    {"broken", "broke", "busted", "damaged", "dead"},
    {"worn", "beat", "beaten", "battered", "scuffed", "rough", "tattered"},
    {"regular", "normal", "ordinary", "standard", "plain", "basic", "common", "generic"},
    {"kid", "child", "children", "youth", "junior"},
    {"anxious", "nervous", "scared", "timid"},
    {"comfortable", "comfy", "cushioned", "soft"},
    {"big", "large", "huge", "giant"},
    {"small", "little", "tiny", "mini"},
    {"cheap", "budget", "inexpensive"},
    {"fast", "quick", "speedy"},
    {"clean", "spotless"},
    {"fine", "ok", "okay", "good", "great"},
    {"dirty", "stained", "stain", "grimy"},
]
SYNONYM_OF = {word: frozenset(group) for group in SYNONYMS for word in group}

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "dozen": 12, "couple": 2, "few": 3,
    "half": 0.5,
}
DECADE_WORDS = {"twenties": 1920, "thirties": 1930, "forties": 1940, "fifties": 1950, "sixties": 1960,
                "seventies": 1970, "eighties": 1980, "nineties": 1990}

# unit word -> (kind, how many of the kind's base unit)
UNITS = {
    "day": ("time", 1 / 365), "days": ("time", 1 / 365), "week": ("time", 1 / 52), "weeks": ("time", 1 / 52),
    "month": ("time", 1 / 12), "months": ("time", 1 / 12), "year": ("time", 1), "years": ("time", 1),
    "yr": ("time", 1), "yrs": ("time", 1), "yo": ("time", 1),
    "hour": ("hours", 1), "hours": ("hours", 1), "hr": ("hours", 1), "hrs": ("hours", 1),
    "foot": ("length", 1), "feet": ("length", 1), "ft": ("length", 1), "inch": ("length", 1 / 12),
    "inches": ("length", 1 / 12),
    "kilo": ("weight", 1), "kilos": ("weight", 1), "kg": ("weight", 1), "kgs": ("weight", 1),
    "pound": ("weight", 0.4536), "pounds": ("weight", 0.4536), "lb": ("weight", 0.4536), "lbs": ("weight", 0.4536),
    "people": ("people", 1), "person": ("people", 1), "persons": ("people", 1),
    "gallon": ("volume", 1), "gallons": ("volume", 1), "liter": ("volume", 0.264), "liters": ("volume", 0.264),
    "mile": ("distance", 1), "miles": ("distance", 1), "km": ("distance", 0.621),
    "count": ("count", 1),          # a bare number when asked "how many" / "how much"
}
MULTIPLIERS = {"hundred": 100, "thousand": 1000, "k": 1000}
BELOW = {"under", "below", "less", "fewer"}
ABOVE = {"over", "above", "more", "plus"}


@dataclass(frozen=True)
class Quantity:
    """A quantity in an answer: its kind (time, kilos, people, count, ...), its value and an optional bound."""
    kind: str
    value: float
    bound: str = ""        # "", "below" or "above": "under ten kilos" covers everything under 10

    def distance(self, typed: float) -> float:
        """How far a typed value is from this one; 0 inside a bound ("under ten kilos" and 7)."""
        if self.bound == "below" and typed < self.value or self.bound == "above" and typed >= self.value:
            return 0.0
        return abs(typed - self.value)


@dataclass(frozen=True)
class Match:
    """Which option the typed answer landed on, and the rule that decided it."""
    option: str | None     # None: can't tell which one
    rule: str              # "yes/no", "number", "words" or "none"


def clean(text: str) -> str:
    """What gets shown and logged: printable ASCII (the game's font), one line, at most MAX_CHARS."""
    text = text.replace("’", "'").replace("‘", "'")
    text = "".join(c if 32 <= ord(c) < 127 else " " for c in text)
    return " ".join(text.split())[:MAX_CHARS]


def tokens(text: str) -> list[str]:
    """Lower-case words and numbers, apostrophes dropped: "She's 12" -> ["shes", "12"]."""
    text = text.lower().replace("'", "").replace("’", "")
    # "20th" stays one token and is not a number ("early 20th century" is not twenty years).
    return re.findall(r"\d+(?:st|nd|rd|th)\b|\d+(?:\.\d+)?s?|[a-z]+", text)


def stem(word: str) -> str:
    """A rough stem: strip one common ending (puppies -> puppy, trained -> train, kids -> kid)."""
    for suffix, keep in (("ies", "y"), ("ing", ""), ("ed", ""), ("ly", ""), ("es", "")):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)] + keep
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]                # kids -> kid (a plain plural, even a short one)
    return word


def _similar(a: str, b: str) -> bool:
    """Same word up to an ending or a typo: a shared start of 5+ letters covering most of the shorter."""
    if a == b:
        return True
    shared = len(_common_prefix(a, b))
    return shared >= 5 and shared >= 0.8 * min(len(a), len(b))


def _common_prefix(a: str, b: str) -> str:
    """The longest start the two words share."""
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


def _same_word(typed: str, option: str) -> bool:
    """A typed word matches an option word: the same up to an ending or typo, or listed as synonyms."""
    if _similar(stem(typed), stem(option)):
        return True
    group = SYNONYM_OF.get(typed) or SYNONYM_OF.get(stem(typed))
    return group is not None and (option in group or stem(option) in group)


# --- rule 1: yes or no ------------------------------------------------------------------------------

def polarity_of_answer(toks: list[str]) -> str:
    """Typed text: only its first word counts ("yes she had her shots" is a yes)."""
    if not toks:
        return ""
    return "yes" if toks[0] in YES else "no" if toks[0] in NO else ""


def polarity_of_option(option: Option) -> str:
    """"yes" or "no" when the option starts with one (or is named so), "no-ish" when it holds a negation, else ""."""
    toks = tokens(option.answer)
    first = polarity_of_answer(toks) or ("yes" if option.id == "yes" else "no" if option.id == "no" else "")
    if first:
        return first
    return "no-ish" if any(t in NEGATIONS for t in toks) else ""


def by_polarity(said: str, options: list[Option]) -> list[Option]:
    """Rule 1: keep the options that agree with a typed yes or no; all of them when nothing was said."""
    if not said:
        return options
    marks = {o.id: polarity_of_option(o) for o in options}
    same = [o for o in options if marks[o.id] == said]
    if same:
        return same
    if said == "no":
        return [o for o in options if marks[o.id] == "no-ish"] or options
    # A yes, and no option starts with one: keep the options that aren't a no ("Spotless." vs "Some
    # stains that won't come out.").
    rest = [o for o in options if not marks[o.id]]
    return rest if len(rest) < len(options) else options


# --- rule 2: numbers --------------------------------------------------------------------------------

def default_unit(question: str) -> str:
    """The unit a bare number means here: years when asked how old, people when asked how many people,
    a plain count when asked how many or how much of something else."""
    toks = tokens(question)
    if "old" in toks or "age" in toks:
        return "years"
    named = next((t for t in toks if t in UNITS), "")
    if named:
        return named
    return "count" if toks[:2] in (["how", "many"], ["how", "much"]) else ""


def quantities(text: str, default: str = "") -> list[Quantity]:
    """Every quantity in the text, in order, with its unit (or `default` for a bare number) and any "under"/"over"
    bound just before it."""
    toks = tokens(text)
    out: list[Quantity] = []
    i = 0
    while i < len(toks):
        value, used = _number_at(toks, i)
        if isinstance(value, Quantity):          # a year or decade: "1910", "the eighties"
            if default == "years":               # asked how old: from the 1920s = about 100 years old
                value = Quantity("time", THIS_YEAR - value.value)
            out.append(value)
        elif value is not None:
            kind, scale = _unit_after(toks[i + used: i + used + 2], default)
            before = set(toks[max(0, i - 2): i])
            bound = "below" if BELOW & before else "above" if ABOVE & before else ""
            out.append(Quantity(kind, value * scale, bound))
        i += used
    return out


def _unit_after(after: list[str], default: str) -> tuple[str, float]:
    """The kind and scale of a number from the words after it: a known unit, else the question's own
    unit, else the next word as a unit we don't know ("256 gigabytes"), else a plain count."""
    unit = next((t for t in after if t in UNITS), default)
    if unit:
        return UNITS[unit]
    if after and after[0].isalpha() and after[0] not in STOPWORDS and after[0] not in NUMBER_WORDS:
        return "per " + stem(after[0]), 1
    return "count", 1


def _calendar(t: str) -> Quantity | None:
    """A year ("1910") or a decade ("1980s", "80s", "eighties", given as its middle year)."""
    if re.fullmatch(r"\d+s", t):
        year = int(t[:-1])
        return Quantity("calendar", (year + 1900 if year < 100 else year) + 5)
    if t in DECADE_WORDS:
        return Quantity("calendar", DECADE_WORDS[t] + 5)
    if re.fullmatch(r"\d{4}", t) and 1000 <= int(t) <= 2100:
        return Quantity("calendar", float(t))
    return None


def _number_at(toks: list[str], i: int) -> tuple[float | Quantity | None, int]:
    """The number starting at toks[i] and how many tokens it took: digits ("4.5", "20 thousand"),
    a year or decade, or words ("a hundred and twenty thousand", "half a dozen")."""
    t = toks[i]
    year = _calendar(t)
    if year is not None:
        return year, 1
    if re.fullmatch(r"\d+(\.\d+)?", t):
        return _words_after(toks, i + 1, float(t), 1)
    if t in ("a", "an"):
        return _after_article(toks, i)
    if t in NUMBER_WORDS:
        return _words_after(toks, i + 1, float(NUMBER_WORDS[t]), 1)
    return None, 1


def _after_article(toks: list[str], i: int) -> tuple[float | None, int]:
    """"a week" is one week, "a few" is three, "a hundred" is a hundred; "a" before anything else is
    not a number."""
    nxt = toks[i + 1] if i + 1 < len(toks) else ""
    if nxt in UNITS:
        return 1.0, 1
    if nxt in ("few", "couple", "half"):
        value, used = _number_at(toks, i + 1)
        return value, used + 1
    if nxt in MULTIPLIERS or nxt == "dozen":
        return _words_after(toks, i + 1, 1.0, 1)
    return None, 1


def _words_after(toks: list[str], j: int, current: float, used: int) -> tuple[float, int]:
    """Keep reading number words after a first number: "thirty three", "two dozen", "four and a half",
    "a hundred and twenty thousand". Returns the value and the tokens used from the first one."""
    total = 0.0
    while j < len(toks) and (step := _number_step(toks, j, current, total)) is not None:
        current, total, n = step
        j, used = j + n, used + n
    if j < len(toks) and toks[j] == "of":
        used += 1                                                   # a couple of gallons
    return total + current, used


def _number_step(toks: list[str], j: int, current: float, total: float) -> tuple[float, float, int] | None:
    """One more word of a number: the new (current, total) and how many tokens it took, or None."""
    return _scale_step(toks[j], current, total) or _joiner_step(toks[j], toks[j + 1: j + 3], current, total)


def _scale_step(w: str, current: float, total: float) -> tuple[float, float, int] | None:
    """A word that multiplies or adds to the number so far."""
    if MULTIPLIERS.get(w, 0) >= 1000:
        return 0.0, total + current * MULTIPLIERS[w], 1            # twenty thousand
    if w in MULTIPLIERS or w == "dozen":
        return current * MULTIPLIERS.get(w, 12), total, 1           # a hundred, two dozen
    if NUMBER_WORDS.get(w, 100) < 100 and w not in ("couple", "few", "half", "dozen"):
        return current + NUMBER_WORDS[w], total, 1                  # thirty three
    return None


def _joiner_step(w: str, nxt: list[str], current: float, total: float) -> tuple[float, float, int] | None:
    """"and" or "a" inside a number, judged by the words after it."""
    if w == "and" and nxt == ["a", "half"]:
        return current + 0.5, total, 3                              # four and a half
    if w == "and" and nxt[:1] and nxt[0] in NUMBER_WORDS:
        return current, total, 1                                    # a hundred and twenty
    if w == "a" and nxt[:1] == ["dozen"] and current == 0.5:
        return current, total, 1                                    # half a dozen
    return None


def by_number(text: str, attr: Attribute, options: list[Option]) -> list[Option]:
    """Rule 2: for the first typed quantity that two or more options also carry, keep the nearest options."""
    unit = default_unit(attr.question)
    typed = quantities(text, unit)
    if not typed:
        return options
    found = {o.id: quantities(o.answer, unit) for o in options}
    for q in typed:
        carrying = [(o, next(x for x in found[o.id] if x.kind == q.kind)) for o in options
                    if any(x.kind == q.kind for x in found[o.id])]
        if len(carrying) < 2:
            continue
        best = min(x.distance(q.value) for _, x in carrying)
        return [o for o, x in carrying if x.distance(q.value) == best]
    return options


# --- rule 3: words ----------------------------------------------------------------------------------

def content(text: str) -> list[str]:
    """The words that carry meaning: no stop words, no numbers."""
    return [t for t in tokens(text) if t not in STOPWORDS and not t[0].isdigit()]


def by_words(text: str, options: list[Option]) -> list[Option]:
    """The options with the best word score; all of them if no word matched."""
    typed = list(dict.fromkeys(content(text)))
    docs = {o.id: (content(o.answer), o.id.split("_")) for o in options}

    def has(word: str, oid: str, part: int) -> bool:
        return any(_same_word(word, w) for w in docs[oid][part])

    scores = {o.id: 0.0 for o in options}
    for word in typed:
        df = sum(1 for o in options if has(word, o.id, 0) or has(word, o.id, 1))
        if df == 0:
            continue
        idf = math.log(1 + len(options) / df)
        for o in options:
            # The answer's own words count fully; the option's name (its id) adds half again.
            scores[o.id] += idf * (has(word, o.id, 0) + 0.5 * has(word, o.id, 1))
    best = max(scores.values())
    if best == 0:
        return options
    return [o for o in options if scores[o.id] == best]


def match(text: str, attr: Attribute) -> Match:
    """Run the three rules in order, each narrowing the options; the first that leaves exactly one decides."""
    options = list(attr.options)
    toks = tokens(text)
    if not toks:
        return Match(None, "none")
    for rule, narrow in (("yes/no", lambda opts: by_polarity(polarity_of_answer(toks), opts)),
                         ("number", lambda opts: by_number(text, attr, opts)),
                         ("words", lambda opts: by_words(text, opts))):
        narrowed = narrow(options)
        if len(narrowed) == 1:
            return Match(narrowed[0].id, rule)
        options = narrowed
    return Match(None, "none")
