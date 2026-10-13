"""Pre-train the GPT on tokenized FineWeb-Edu (data/processed/pretrain/*.bin).

    python -m sellanything_ml.llm.pretrain --preset small --tokens 300_000_000

Resumable (rerun the same command). Every --eval-every steps: validation loss on fixed batches,
a short generated sample, and a checkpoint. Metrics go to <run-dir>/metrics.jsonl.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from .model import GPT, GPTConfig
from .prepare import OUT_DIR as DATA_DIR
from .tokenizer import ML_DIR, TOKENIZER_DIR, Tokenizer

PRESETS = {
    "tiny": dict(d_model=256, n_layers=4, n_heads=4, n_kv_heads=2),
    "mini": dict(d_model=384, n_layers=6, n_heads=6, n_kv_heads=3),
    "small": dict(d_model=512, n_layers=8, n_heads=8, n_kv_heads=4),
    "medium": dict(d_model=640, n_layers=10, n_heads=10, n_kv_heads=5),
    "large": dict(d_model=768, n_layers=9, n_heads=12, n_kv_heads=6),      # 64.7M: 20 tokens a parameter on 20 shards
}
MAX_SEQ_LEN = 1024           # positions the model is built for (RoPE tables); training windows are --block long
LOG_EVERY = 25               # steps between progress lines
GRAD_CLIP = 1.0              # gradient norm clip
SAMPLE_PROMPT = "The best way to learn a new skill is"
SAMPLE = {"max_new_tokens": 40, "temperature": 0.8, "top_k": 50}     # the text written at every evaluation


def pick_device() -> torch.device:
    """CUDA if there is one, else Apple's MPS, else the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Batches:
    """Random windows from a memory-mapped token file (the nanoGPT get_batch)."""

    def __init__(self, path: Path, block: int, batch: int, device: torch.device):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block, self.batch, self.device = block, batch, device

    def get(self, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """`batch` random windows of `block` tokens as inputs, and the same windows one token later as targets."""
        starts = rng.integers(0, len(self.data) - self.block - 1, self.batch)
        chunk = np.stack([self.data[s:s + self.block + 1] for s in starts]).astype(np.int64)
        t = torch.from_numpy(chunk).to(self.device)
        return t[:, :-1], t[:, 1:]


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    """Linear warm-up, then cosine decay to 10% of peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model: GPT, val: Batches, n_batches: int = 20) -> float:
    """Mean validation loss over n_batches windows, the same windows every time."""
    model.eval()
    rng = np.random.default_rng(1234)            # same batches every time, so curves are comparable
    losses = []
    for _ in range(n_batches):
        x, y = val.get(rng)
        losses.append(model(x, y)[1].item())
    model.train()
    return float(np.mean(losses))


def make_optimizer(model: GPT, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with weight decay on the matrices only, not on the norm gains."""
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]   # norms
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), eps=1e-8)


def parse_args() -> argparse.Namespace:
    """Command-line options (see --help)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=PRESETS, default="small")
    parser.add_argument("--tokenizer", default="sp8192")
    parser.add_argument("--block", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    parser.add_argument("--tokens", type=float, default=300e6, help="training tokens in total")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=0, help="stop early (benchmarks)")
    parser.add_argument("--run-dir", type=Path, default=ML_DIR / "runs" / "pretrain")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="folder with train.bin and val.bin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16",
                        help="autocast precision for the forward pass (weights stay float32)")
    return parser.parse_args()


def resume(model: GPT, optimizer: torch.optim.Optimizer, last: Path, device: torch.device) -> int:
    """Load the last checkpoint if the run was started before; returns the step to continue from."""
    if not last.exists():
        return 0
    state = torch.load(last, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    print(f"resumed at step {state['step']}", flush=True)
    return state["step"]


def train_step(model: GPT, optimizer: torch.optim.Optimizer, next_batch: Callable[[], tuple[torch.Tensor, torch.Tensor]],
               accum: int, autocast: Callable[[], contextlib.AbstractContextManager]) -> torch.Tensor:
    """One optimizer step over `accum` micro-batches; returns the last micro-batch's (scaled) loss."""
    for _ in range(accum):
        x, y = next_batch()
        with autocast():
            loss = model(x, y)[1] / accum
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss


def main() -> None:
    """Build the model from a preset, resume if a checkpoint exists, train to --tokens (or --max-steps),
    evaluating, writing a sample and saving every --eval-every steps."""
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()
    tok = Tokenizer(TOKENIZER_DIR / f"{args.tokenizer}.model")
    cfg = GPTConfig(vocab_size=tok.vocab_size, max_seq_len=MAX_SEQ_LEN, **PRESETS[args.preset])
    model = GPT(cfg).to(device)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)

    tokens_per_step = args.batch * args.block * args.accum
    total_steps = int(args.tokens // tokens_per_step)
    train = Batches(args.data_dir / "train.bin", args.block, args.batch, device)
    val = Batches(args.data_dir / "val.bin", args.block, args.batch, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    last = args.run_dir / "last.pt"
    step = resume(model, optimizer, last, device)
    print(f"model {model.num_params():,} params ({args.preset}), {total_steps:,} steps × "
          f"{tokens_per_step:,} tokens = {total_steps * tokens_per_step / 1e6:.0f}M tokens; "
          f"train.bin has {len(train.data) / 1e6:.0f}M tokens; device {device}", flush=True)

    def autocast() -> contextlib.AbstractContextManager:
        if args.dtype == "float32":
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=getattr(torch, args.dtype))

    rng = np.random.default_rng(args.seed + step)      # a resumed run does not replay the same batches
    prompt = torch.tensor([tok.encode(SAMPLE_PROMPT)], device=device)
    t0, tokens_seen = time.time(), 0
    end = min(total_steps, args.max_steps) if args.max_steps else total_steps
    while step < end:
        lr = lr_at(step, total_steps, args.lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss = train_step(model, optimizer, lambda: train.get(rng), args.accum, autocast)
        step += 1
        tokens_seen += tokens_per_step
        if step % LOG_EVERY == 0:
            rate = tokens_seen / (time.time() - t0)
            print(f"step {step} loss {loss.item() * args.accum:.3f} lr {lr:.2e} {rate:,.0f} tok/s", flush=True)
        if step % args.eval_every == 0 or step == end:
            sample = tok.decode(model.generate(prompt, **SAMPLE)[0].tolist())
            record = {"step": step, "tokens": step * tokens_per_step, "val_loss": evaluate(model, val),
                      "lr": lr, "tok_per_s": tokens_seen / (time.time() - t0), "sample": sample}
            with open(args.run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "config": cfg.to_dict(), "tokenizer": args.tokenizer}, last)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "tokenizer": args.tokenizer,
                "step": step}, args.run_dir / "final.pt")

if __name__ == "__main__":
    main()
