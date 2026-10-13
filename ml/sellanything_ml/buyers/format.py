"""Prompt format for the buyer model (docs/buyer-ai.md §7).

Training data and inference both go through these functions, so the model never sees a format at
play time that differs from what it was trained on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ITEM, ATTRS, PERSONA, LOG = "<|item|>", "<|attrs|>", "<|persona|>", "<|log|>"
ASK, REACT, REFUSE = "<|ask|>", "<|react|>", "<|refuse|>"
INTEREST, END = "<|interest|>", "<|end|>"
SPECIAL_TOKENS = [ITEM, ATTRS, PERSONA, LOG, ASK, REACT, REFUSE, INTEREST, END]

# Tokens added after pre-training (design-v2 §6.4). SentencePiece has no free ids, so these get ids
# after its vocabulary (llm/tokenizer.py) and the embedding grows when a pre-trained model is
# fine-tuned with them (llm/model.py grow_vocab). Only <|parse|> and <|said|> are used so far; the
# rest are reserved so later tasks need no second growth.
PARSE, SAID = "<|parse|>", "<|said|>"
# B plan (prompt format 4, buyers/sales_sft.py): the first two spare slots became the opening and the imagined scene.
# Renaming a spare keeps its id, and no model has been trained on the spares.
OPEN, IMAGINE = "<|open|>", "<|imagine|>"
EXTRA_TOKENS = [PARSE, SAID, "<|profile|>", "<|which|>", "<|answer|>", "<|weights|>", OPEN, IMAGINE] + \
               [f"<|spare{i}|>" for i in range(2, 8)]


@dataclass(frozen=True)
class BuyerView:
    """What one buyer knows at this moment: what is for sale (and, sometimes, which attributes it
    has), who they are, and everything said so far."""

    noun: str
    attrs: list[str] | None
    persona: str
    log: list[str]
    names: tuple[str, ...] = ()      # how lines name the item (item_names); empty: from noun alone
    cares: tuple[str, ...] = ()      # tags this buyer weighs (their traits); the reasons they may give

    def render(self) -> str:
        """The shared prompt prefix."""
        parts = [f"{ITEM}{self.noun}{END}"]
        if self.attrs:
            parts.append(f"{ATTRS}{', '.join(self.attrs)}{END}")
        parts.append(f"{PERSONA}{self.persona}{END}")
        parts.append(f"{LOG}{chr(10).join(self.log)}{END}")
        return "".join(parts)


# Prompt format 1 (data v1, v2): the ask prompt names only the attribute, and the model has to
# come up with the question. Format 2 (v3): the engine's question is in the prompt and the model
# says it in the buyer's voice — the only way it can ask well about items it never saw.
# Format 3 (v4): as 2, but the model never spells the item. Wherever a line names it, the model
# writes ITEM_SLOT and the engine fills the name in (fill_slot): a small model garbles names of
# items it never saw ("saxophone" → "saxmalone"), and the engine knows what is for sale anyway.
PROMPT_FORMAT = 3
ITEM_SLOT = "{item}"


def ask_prompt(view: BuyerView, attribute: str, question: str | None = None) -> str:
    """With `question` (format 2) the prompt ends `<|ask|>type: Is it digital?<|end|>`."""
    asked = f"{attribute}: {question}" if question else attribute
    return view.render() + f"{ASK}{asked}{END}"


def react_prompt(view: BuyerView) -> str:
    """The buyer's view of the round so far, ending `<|react|>`: the model writes its reaction next."""
    return view.render() + REACT


def refuse_prompt(noun: str, persona: str) -> str:
    """An item nobody buys and one buyer, ending `<|refuse|>`: the model writes why this buyer won't."""
    return f"{ITEM}{noun}{END}{PERSONA}{persona}{END}{REFUSE}"


def ask_target(name: str, line: str) -> str:
    """What the model learns to write after `<|ask|>`: `Name: line<|end|>`."""
    return f"{name}: {line}{END}"


def react_target(name: str, line: str | None, delta: int) -> str:
    """`Name: line<|interest|>+1<|end|>`, or only `<|interest|>0<|end|>` for a buyer who stays silent."""
    speech = f"{name}: {line}" if line else ""
    return f"{speech}{INTEREST}{delta:+d}{END}"


refuse_target = ask_target    # a refusal is written exactly like a question: `Name: line<|end|>`


# parse (design-v2 §6.3): which option did the player mean? The options are lettered in the order
# given (training shuffles it), so the answer is one letter whatever the item, or "none".
LETTERS = "ABCDEFGH"
NO_OPTION = "none"


def parse_prompt(noun: str, question: str, options: list[str], said: str) -> str:
    """<|item|>cat<|end|><|parse|>What breed is the cat?
    A. A purebred Maine Coon, with papers.
    B. A regular mixed-breed cat.<|end|><|said|>british shorthair<|end|>"""
    listed = "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(options))
    return f"{ITEM}{noun}{END}{PARSE}{question}\n{listed}{END}{SAID}{' '.join(said.split())}{END}"


def parse_target(index: int | None) -> str:
    """The option's letter ("B<|end|>"), or "none<|end|>" when the typed answer matches no option."""
    return f"{NO_OPTION if index is None else LETTERS[index]}{END}"


def read_parse(text: str, n_options: int) -> int | None | bool:
    """The option index from a parse output, None for "none", False if the output is malformed."""
    out = text.strip().removesuffix(END).strip()
    if out == NO_OPTION:
        return None
    if len(out) == 1 and out in LETTERS[:n_options]:
        return LETTERS.index(out)
    return False


_REACT_RE = re.compile(r"^(?:(?P<name>[^:<]+): (?P<line>.*?))?<\|interest\|>(?P<delta>[+-][0-2])<\|end\|>$", re.S)
_LINE_RE = re.compile(r"^(?P<name>[^:<]+): (?P<line>.+?)<\|end\|>$", re.S)


def parse_react(text: str) -> tuple[str | None, str | None, int] | None:
    """(name, line, Δ) from a react output, or None if it doesn't follow the format."""
    m = _REACT_RE.match(text.strip())
    if not m:
        return None
    return m["name"], m["line"], int(m["delta"])


def parse_line(text: str) -> tuple[str, str] | None:
    """(name, line) from an ask or refuse output."""
    m = _LINE_RE.match(text.strip())
    return (m["name"], m["line"]) if m else None


# Words that carry no topic: "How old is it?" is about "old" and nothing else.
FUNCTION_WORDS = {"a", "an", "the", "is", "it", "its", "it's", "are", "was", "be", "do", "does", "did",
                  "has", "have", "how", "what", "which", "any", "or", "and", "of", "to", "in", "on",
                  "for", "with", "you", "your", "this", "that", "there", "can"}


def _content_words(text: str) -> set[str]:
    """Lower-case words of the text without punctuation and without FUNCTION_WORDS."""
    return {w for w in (x.strip("?,.!").lower() for x in text.split()) if w and w not in FUNCTION_WORDS}


def on_topic(line: str, question: str, noun: str) -> bool:
    """Does a buyer's line still ask `question`? At least half of its topic words, and the item's
    name whenever the question uses it ("what kind of kittery is it?" is not about a knife).
    The guardrail and the evaluation both use this, so they agree on what "on topic" means."""
    said = line.lower()
    words = _content_words(question)
    if not words:                       # nothing but function words: the question itself must be there
        return question.lower().rstrip("?") in said
    named = [w for w in noun.lower().split() if w in words]
    return sum(w in said for w in words) / len(words) >= 0.5 and all(w in said for w in named)


def item_names(item_key: str, noun: str) -> list[str]:
    """Ways a line can name the item: key, noun, and their last words ("alarm clock" → "clock").
    Longest first, so "alarm clock" is matched before "clock". Short last words ("saw") are left
    out: they're too likely to mean something else."""
    names = {item_key, noun}
    names |= {n.split()[-1] for n in (item_key, noun) if len(n.split()[-1]) >= 4}
    return sorted(names, key=len, reverse=True)


def _names_pattern(names: list[str]) -> re.Pattern[str]:
    """One regex matching any of the names as a whole word, case-insensitive."""
    return re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)


def to_slot(text: str, names: list[str]) -> str:
    """Every whole-word mention of the item becomes ITEM_SLOT (articles are left for fill_slot)."""
    return _names_pattern(names).sub(ITEM_SLOT, text)


def fill_slot(text: str, noun: str) -> str:
    """ITEM_SLOT back to the item's name, with "a"/"an" agreeing and a capital at a sentence start."""
    article = "an" if noun[:1].lower() in "aeiou" else "a"
    text = re.sub(r"\b([Aa])n? " + re.escape(ITEM_SLOT),
                  lambda m: (article.capitalize() if m.group(1) == "A" else article) + " " + noun, text)
    text = re.sub(r"(^|[.!?]\s+)" + re.escape(ITEM_SLOT), lambda m: m.group(1) + noun[:1].upper() + noun[1:], text)
    return text.replace(ITEM_SLOT, noun)
