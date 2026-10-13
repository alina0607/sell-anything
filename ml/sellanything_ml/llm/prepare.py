"""Tokenize pre-training text into flat uint16 arrays (nanoGPT's train.bin format).

    python -m sellanything_ml.llm.prepare --tokenizer sp8192

FineWeb-Edu shards 0..n-2 become train.bin, the last shard becomes val.bin. Documents are
separated by <|end|>; paragraphs inside a document by the newline piece.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .tokenizer import FINEWEB_DIR, ML_DIR, TOKENIZER_DIR, Tokenizer

OUT_DIR = ML_DIR / "data" / "processed" / "pretrain"


def tokenize_shard(tok: Tokenizer, shard: Path, max_docs: int | None = None) -> np.ndarray:
    """One FineWeb-Edu shard as one flat array of token ids: each document's lines joined by the newline piece,
    every document ended by <|end|>."""
    docs = pq.read_table(shard, columns=["text"]).column("text").to_pylist()[:max_docs]
    newline = tok.sp.piece_to_id("\n")
    lines_per_doc = [doc.split("\n") for doc in docs]
    flat = [line for lines in lines_per_doc for line in lines]
    encoded = tok.sp.encode(flat, num_threads=8)          # one batched call: SentencePiece's C++ threads do the work
    out: list[int] = []
    i = 0
    for lines in lines_per_doc:
        for j in range(len(lines)):
            if j:
                out.append(newline)
            out.extend(encoded[i])
            i += 1
        out.append(tok.eos_id)
    return np.array(out, dtype=np.uint16)


def main() -> None:
    """Tokenize every shard but the last into train.bin and the last one's first --val-docs documents into val.bin."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", default="sp8192")
    parser.add_argument("--val-docs", type=int, default=20_000)
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output folder (train.bin, val.bin, meta.json)")
    args = parser.parse_args()
    out_dir = args.out

    tok = Tokenizer(TOKENIZER_DIR / f"{args.tokenizer}.model")
    assert tok.vocab_size <= 65535, "uint16 storage"       # every id has to fit in the .bin files' 16 bits
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(FINEWEB_DIR.glob("shard_*.parquet"))
    stats = {}
    t = time.time()
    with open(out_dir / "train.bin", "wb") as f:
        total = 0
        for shard in shards[:-1]:
            ids = tokenize_shard(tok, shard)
            ids.tofile(f)
            total += len(ids)
            print(f"{shard.name}: {len(ids):,} tokens ({time.time() - t:.0f}s)", flush=True)
    stats["train_tokens"] = total
    val = tokenize_shard(tok, shards[-1], args.val_docs)
    val.tofile(out_dir / "val.bin")
    stats["val_tokens"] = len(val)
    stats["tokenizer"] = args.tokenizer
    (out_dir / "meta.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
