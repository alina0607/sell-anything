"""The parse task's prompt, its answer and the tokens added after pre-training (design-v2 §6.3, §6.4)."""

import pytest

from sellanything_ml.buyers import format as fmt
from sellanything_ml.llm.tokenizer import TOKENIZER_DIR, Tokenizer


def test_parse_prompt_letters_options_and_flattens_the_typed_text():
    p = fmt.parse_prompt("cat", "What breed is the cat?", ["Purebred.", "Mixed."], "british\n  shorthair")
    assert p == ("<|item|>cat<|end|><|parse|>What breed is the cat?\nA. Purebred.\nB. Mixed.<|end|>"
                 "<|said|>british shorthair<|end|>")
    assert fmt.parse_target(1) == "B<|end|>" and fmt.parse_target(None) == "none<|end|>"


def test_read_parse_knows_letters_none_and_garbage():
    assert fmt.read_parse("B<|end|>", 2) == 1
    assert fmt.read_parse(" none <|end|>", 2) is None
    assert fmt.read_parse("C<|end|>", 2) is False          # no third option
    assert fmt.read_parse("Maya: ok<|end|>", 2) is False


@pytest.mark.skipif(not (TOKENIZER_DIR / "sp8192.model").exists(),
                    reason="the trained tokenizer is not in the repository (checkpoints/ is ignored)")
def test_added_tokens_are_single_ids_after_sentencepiece_and_round_trip():
    tok = Tokenizer(TOKENIZER_DIR / "sp8192.model")
    text = fmt.parse_prompt("dog", "How old is the dog?", ["A puppy.", "Two."], "like 3 months") + fmt.parse_target(0)
    ids = tok.encode(text)
    assert [i for i in ids if i >= tok.sp_size] == [tok.sp_size + fmt.EXTRA_TOKENS.index(t) for t in (fmt.PARSE, fmt.SAID)]
    assert tok.decode(ids) == text
    old = "<|item|>dog<|end|><|log|>Maya: Hi\nYou: Two years.<|end|><|react|>"
    assert tok.encode(old) == tok._encode_sp(old)              # old prompts encode exactly as before


def test_the_first_option_is_an_answer_not_a_format_error(monkeypatch):
    # read_parse returns 0 for "A"; `0 in (None, False)` is True in Python, which once scored every A as wrong
    from sellanything_ml.buyers.schema import load_items
    from sellanything_ml.llm import evaluate
    attr = load_items()["bird"].attributes["species"]
    monkeypatch.setattr(evaluate, "complete", lambda *a, **k: "A<|end|>")
    assert evaluate.parse_choice(None, None, "pet bird", attr, "budgie") == attr.options[0].id
    ex = {"item": "bird", "attr": "species", "option": "budgie", "said": "budgie", "prompt": "",
          "order": ["budgie", "cockatiel", "canary", "lovebirds"], "typo": False}
    assert evaluate.parse_report(object(), None, [ex])["model all"] == {"n": 1, "right": 1.0}
