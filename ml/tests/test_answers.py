"""Typed answers (buyers/answers.py): the parsing rules, the hand-written answer sets, and a round trip
over every option in the catalog."""

from pathlib import Path

import pytest

from sellanything_ml.buyers.answers import Quantity, clean, match, quantities
from sellanything_ml.buyers.schema import SELLABLE, load_items, load_labels

FIXTURES = Path(__file__).parent / "fixtures"
ITEMS = load_items()


def _rows(name):
    for line in (FIXTURES / f"typed_answers_{name}.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            yield line.split("\t")


@pytest.mark.parametrize("text, default, want", [
    ("A hundred and twenty thousand miles.", "", Quantity("distance", 120000)),
    ("Half a dozen.", "count", Quantity("count", 6)),
    ("A box of two dozen.", "count", Quantity("count", 24)),
    ("Four and a half feet.", "", Quantity("length", 4.5)),
    ("a couple of gallons", "", Quantity("volume", 2)),
    ("Small, under ten kilos.", "", Quantity("weight", 10, "below")),
    ("A three-month-old kitten.", "years", Quantity("time", 0.25)),
    ("Just one.", "", Quantity("count", 1)),
])
def test_quantities(text, default, want):
    assert quantities(text, default) == [want]


def test_years_on_an_age_question_mean_years_ago():
    [q] = quantities("From the eighties.", "years")
    assert q.kind == "time" and 35 < q.value < 50


def test_an_ordinal_is_not_a_number():
    # "20th" once read as "20 years old" and picked the wrong option
    assert quantities("early 20th century", "years") == []


def test_clean_keeps_one_short_printable_line():
    assert clean("  she’s\n  12 \x00 ") == "she's 12"
    assert len(clean("x" * 500)) == 120


def test_each_rule_is_reported():
    age, litter = ITEMS["cat"].attributes["age"], ITEMS["cat"].attributes["litter"]
    assert match("shes like 12", age).option == "senior" and match("shes like 12", age).rule == "number"
    assert match("yep", litter).option == "yes" and match("yep", litter).rule == "yes/no"
    assert match("a three month old kitten", age).option == "kitten"
    assert match("asdf", age).option is None and match("", age).rule == "none"


@pytest.mark.parametrize("name, at_least", [("dev", 0.97), ("test", 0.97), ("test2", 0.95)])
def test_hand_written_answers(name, at_least):
    """Regression gate. The honest numbers are the first runs, before any tuning (docs/buyer-ai.md):
    test 97/99, test2 59/61 (one wrong pick: 'bike helmet' -> the kids' bike helmet)."""
    rows = list(_rows(name))
    got = [(match(typed, ITEMS[item].attributes[attr]).option or "-", want) for item, attr, typed, want in rows]
    right = sum(g == w for g, w in got)
    wrong_picks = sum(g != w and g != "-" for g, w in got)
    assert right / len(rows) >= at_least
    assert wrong_picks <= (1 if name == "test2" else 0)


def test_every_option_text_matches_itself_and_never_another():
    labels = load_labels()
    wrong = []
    for key, item in ITEMS.items():
        if labels.get(key) not in SELLABLE:
            continue
        for attr in item.attributes.values():
            for option in attr.options:
                found = match(option.answer, attr).option
                if found is not None and found != option.id:
                    wrong.append((key, attr.name, option.answer, found))
    assert wrong == []
