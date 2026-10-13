"""format.on_topic against hand-labeled lines (tests/fixtures/on_topic_gold.jsonl).

The rule decides which model lines reach the game and what the evaluation reports as "on topic",
so it is checked against labels that did not come from the rule: real model outputs read one by
one, including look-alikes that must be rejected ("what kind of kittery is it?" for a knife).
Add a row whenever a line is judged wrongly in play; a disagreement is a bug in the rule or in
the label, and the row's `why` says which reading the label is based on.
"""

import json
from pathlib import Path

from sellanything_ml.buyers.format import on_topic

ROWS = [json.loads(line) for line in (Path(__file__).parent / "fixtures" / "on_topic_gold.jsonl").open()]


def test_the_gold_set_can_fail_a_bad_rule():
    # A rule that always says yes (or always no) must not pass: both labels are well represented.
    labels = [r["on_topic"] for r in ROWS]
    assert sum(labels) >= 5 and labels.count(False) >= 5


def test_rule_agrees_with_every_hand_label():
    wrong = [f'{r["on_topic"]!s:>5} expected for {r["line"]!r} (asked: {r["question"]!r}; {r["why"]})'
             for r in ROWS if on_topic(r["line"], r["question"], r["noun"]) != r["on_topic"]]
    assert not wrong, "\n".join(wrong)


def test_the_gold_set_rejects_the_rule_it_replaced():
    # The rule before 2026-10-03: half the words longer than three letters, item name not checked.
    # It let through 12 of these 24 lines, all look-alikes. If edits to the gold set make the old
    # rule pass, the set has lost the cases it exists for.
    def old_rule(line, question, noun):
        words = {w.strip("?,.").lower() for w in question.split() if len(w) > 3}
        return not words or sum(w in line.lower() for w in words) / len(words) >= 0.5

    assert sum(old_rule(r["line"], r["question"], r["noun"]) != r["on_topic"] for r in ROWS) >= 10
