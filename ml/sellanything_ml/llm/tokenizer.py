"""SentencePiece BPE tokenizer for the buyer model (docs/plan.md D7).

    python -m sellanything_ml.llm.tokenizer --vocab 8192

Trained on FineWeb-Edu text plus buyer dialogues so that both general English and the game's
format compress well. The chat special tokens and the newline are single, never-split pieces.
Normalization is turned off so that decode(encode(text)) == text exactly.

Tokens added after pre-training (format.EXTRA_TOKENS) are not SentencePiece pieces: they get the ids
right after its vocabulary, so every id SentencePiece produces keeps its meaning and a pre-trained
model only needs its embedding grown (model.grow_vocab).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm

from ..buyers.datagen import BUYERS_DIR
from ..buyers.format import EXTRA_TOKENS, SPECIAL_TOKENS

ML_DIR = Path(__file__).resolve().parents[2]
FINEWEB_DIR = ML_DIR / "data" / "raw" / "fineweb_edu"
TOKENIZER_DIR = ML_DIR / "checkpoints" / "tokenizer"
NEWLINE = "\n"


class Tokenizer:
    """A trained SentencePiece model plus the tokens added after pre-training (format.EXTRA_TOKENS), which get
    the ids right after SentencePiece's own."""

    def __init__(self, model_file: Path | str):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_file))
        self.sp_size = self.sp.get_piece_size()
        self.extra = {tok: self.sp_size + i for i, tok in enumerate(EXTRA_TOKENS)}
        self.extra_text = {i: tok for tok, i in self.extra.items()}
        self.vocab_size = self.sp_size + len(EXTRA_TOKENS)
        self.special = {tok: self.sp.piece_to_id(tok) for tok in SPECIAL_TOKENS}
        self.eos_id = self.special["<|end|>"]
        self.pad_id = self.sp.pad_id()
        self._extra_re = re.compile("(" + "|".join(re.escape(t) for t in EXTRA_TOKENS) + ")")

    def encode(self, text: str) -> list[int]:
        """Text → token ids. The added tokens (format.EXTRA_TOKENS) are cut out first and get their own ids;
        everything between them goes through SentencePiece."""
        ids: list[int] = []
        for part in self._extra_re.split(text):
            if part in self.extra:
                ids.append(self.extra[part])
            elif part:
                ids.extend(self._encode_sp(part))
        return ids

    def _encode_sp(self, text: str) -> list[int]:
        """SentencePiece ids for text without added tokens, keeping its newlines."""
        # Encode line by line and join with the newline piece: SentencePiece would otherwise treat
        # "\n" as whitespace.
        ids: list[int] = []
        newline = self.sp.piece_to_id(NEWLINE)
        for i, line in enumerate(text.split(NEWLINE)):
            if i:
                ids.append(newline)
            if line:
                ids.extend(self.sp.encode(line))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Token ids → text, the exact inverse of `encode` (normalization is off, so round trips are exact)."""
        out, run = [], []
        for i in ids:
            if i in self.extra_text:
                out.append(self.sp.decode(run))
                out.append(self.extra_text[i])
                run = []
            else:
                run.append(i)
        out.append(self.sp.decode(run))
        return "".join(out)


def iter_fineweb_paragraphs(shards: list[Path]) -> Iterator[str]:
    """Every non-empty paragraph of every FineWeb-Edu document in the shards, in order."""
    for shard in shards:
        table = pq.read_table(shard, columns=["text"])
        for doc in table.column("text").to_pylist():
            for para in doc.split("\n"):
                para = para.strip()
                if para:
                    yield para


def write_corpus(path: Path, fineweb_chars: int, buyer_chars: int) -> dict[str, int]:
    """One sentence per line: FineWeb paragraphs, then lines from buyer prompts and targets."""
    stats = {"fineweb_chars": 0, "buyer_chars": 0}
    with open(path, "w", encoding="utf-8") as f:
        shards = sorted(FINEWEB_DIR.glob("shard_*.parquet"))[:-1]   # last shard held out for eval
        for para in iter_fineweb_paragraphs(shards):
            if stats["fineweb_chars"] >= fineweb_chars:
                break
            f.write(para + "\n")
            stats["fineweb_chars"] += len(para)
        with open(BUYERS_DIR / "train.jsonl", encoding="utf-8") as src:
            for row in src:
                if stats["buyer_chars"] >= buyer_chars:
                    break
                ex = json.loads(row)
                for line in (ex["prompt"] + ex["target"]).split("\n"):
                    if line.strip():
                        f.write(line + "\n")
                        stats["buyer_chars"] += len(line)
    return stats


def train(vocab: int, corpus: Path, out_dir: Path) -> Path:
    """Train a BPE SentencePiece model of `vocab` pieces on the corpus; returns the .model path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"sp{vocab}"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=vocab,
        character_coverage=0.9995,
        byte_fallback=True,                       # any character can be encoded
        split_digits=True,                        # prices: one digit per piece
        user_defined_symbols=SPECIAL_TOKENS + [NEWLINE],
        normalization_rule_name="identity",       # no NFKC rewriting, exact round trips
        remove_extra_whitespaces=False,
        add_dummy_prefix=False,                   # we encode line by line; no invented spaces
        pad_id=0, unk_id=1, bos_id=-1, eos_id=-1,
        input_sentence_size=3_000_000,
        shuffle_input_sentence=True,
        max_sentence_length=8192,
        num_threads=8,
        minloglevel=2,
    )
    return prefix.with_suffix(".model")


def main() -> None:
    """Write the corpus once (it is reused), then train one tokenizer per --vocab size."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vocab", type=int, nargs="+", default=[8192])
    parser.add_argument("--fineweb-chars", type=int, default=150_000_000)
    parser.add_argument("--buyer-chars", type=int, default=20_000_000)
    parser.add_argument("--out", type=Path, default=TOKENIZER_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    corpus = args.out / "corpus.txt"
    if not corpus.exists():
        t = time.time()
        print("corpus:", write_corpus(corpus, args.fineweb_chars, args.buyer_chars),
              f"({time.time() - t:.0f}s)", flush=True)
    for vocab in args.vocab:
        t = time.time()
        model = train(vocab, corpus, args.out)
        print(f"trained {model.name} in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
