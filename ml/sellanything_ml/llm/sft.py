"""Supervised fine-tuning on buyer dialogues (loss only on the buyer's reply).

    python -m sellanything_ml.llm.sft --init runs/pretrain/final.pt --epochs 2

Each example is prompt + target; the loss mask is 1 only on target tokens, so the model learns
to produce replies, not to predict the prompt. Validation: masked loss every --eval-every steps,
and task metrics (evaluate.py) on held-out items at the end.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..buyers.datagen import BUYERS_DIR, is_current, prompt_format_of
from . import evaluate as ev
from .model import GPT, GPTConfig, grow_vocab
from .pretrain import make_optimizer, pick_device
from .tokenizer import ML_DIR, TOKENIZER_DIR, Tokenizer

DATA = BUYERS_DIR
WARMUP_STEPS = 100
MIN_LR_SHARE = 0.1          # the cosine schedule ends at this share of the peak learning rate
GRAD_CLIP = 1.0
LOG_EVERY = 50
VAL_LIMIT = 4000            # validation examples read per dataset
VAL_BATCH, VAL_BATCHES = 16, 120     # 1,920 examples like 64 x 30, in less memory
FINAL_EVAL_N = 400
Encoded = tuple[list[int], int]      # token ids of prompt + target, and how many of them are the prompt


def encode_examples(tok: Tokenizer, path: Path, max_len: int, limit: int | None = None) -> tuple[list[Encoded], int]:
    """[(ids, n_prompt_tokens)] for the first `limit` lines (all if None), skipping anything longer than max_len;
    also how many were skipped."""
    out: list[Encoded] = []
    skipped = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            ex = json.loads(line)
            p, t = tok.encode(ex["prompt"]), tok.encode(ex["target"])
            if len(p) + len(t) > max_len:
                skipped += 1
                continue
            out.append((p + t, len(p)))
    return out, skipped


PAD_TO = 64
# Slowdown watchdog (memory: GPU loops stop themselves instead of wasting the GPU): the median step time of the
# last WATCH_WINDOW steps against that of the first steps after warm-up.
WATCH_WINDOW, WATCH_SKIP, SLOWDOWN_FACTOR = 50, 10, 2.5


def slowed_down(times: list[float]) -> bool:
    """Step times so far → True when the recent median is SLOWDOWN_FACTOR times the early one."""
    early = times[WATCH_SKIP:WATCH_SKIP + WATCH_WINDOW]
    if len(early) < WATCH_WINDOW or len(times) < WATCH_SKIP + 2 * WATCH_WINDOW:
        return False
    recent = sorted(times[-WATCH_WINDOW:])[WATCH_WINDOW // 2]
    return recent > SLOWDOWN_FACTOR * sorted(early)[WATCH_WINDOW // 2]


def collate(batch: list[Encoded], pad_id: int, device: torch.device,
            per_example: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inputs, targets and the loss mask. per_example: each example's target tokens share a weight of 1,
    so every example counts the same whatever its length; otherwise every target token counts 1."""
    # Round the length up to a multiple of PAD_TO: a handful of shapes instead of hundreds.
    # With one shape per batch, MPS kept allocating until it swapped (20 GB for a 13M model).
    T = math.ceil((max(len(ids) for ids, _ in batch) - 1) / PAD_TO) * PAD_TO
    x = torch.full((len(batch), T), pad_id, dtype=torch.long)
    y = torch.full((len(batch), T), pad_id, dtype=torch.long)
    mask = torch.zeros(len(batch), T)
    for i, (ids, n_prompt) in enumerate(batch):
        seq = torch.tensor(ids)
        n = len(ids) - 1
        x[i, :n], y[i, :n] = seq[:-1], seq[1:]
        mask[i, n_prompt - 1:n] = 1 / (n - n_prompt + 1) if per_example else 1   # target tokens only (incl. <|end|>)
    return x.to(device), y.to(device), mask.to(device)


@torch.no_grad()
def val_loss(model: GPT, data: list[Encoded], pad_id: int, device: torch.device, batch: int = 64,
             n_batches: int = 30) -> float:
    """Mean masked loss over n_batches random batches of the validation examples (always the same: seed 0)."""
    model.eval()
    rng = random.Random(0)
    losses = []
    for _ in range(n_batches):
        x, y, m = collate(rng.sample(data, batch), pad_id, device, per_example=False)
        losses.append(model(x, y, m)[1].item())
    model.train()
    return sum(losses) / len(losses)


def parse_args() -> argparse.Namespace:
    """Command-line options (see --help)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--init", type=Path, required=True, help="pre-trained checkpoint")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-len", type=int, default=768)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--run-dir", type=Path, default=ML_DIR / "runs" / "sft")
    parser.add_argument("--data", type=Path, default=DATA, help="dataset dir (train.jsonl, val.jsonl)")
    parser.add_argument("--extra-data", type=Path, action="append", default=[],
                        help="additional compatible dataset dir; may be repeated")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--loss", choices=["per-token", "per-example"], default="per-token",
                        help="per-example: a 2-token parse answer weighs as much as a 30-token reply (data v6)")
    parser.add_argument("--max-minutes", type=float, default=0,
                        help="stop after this many minutes, evaluating and saving first (0 = no limit; paid GPUs)")
    parser.add_argument("--start-step", type=int, default=0,
                        help="continue the schedule of an interrupted run (with --init its checkpoint)")
    return parser.parse_args()


def dataset_format(data_dirs: list[Path]) -> tuple[int, list[str]]:
    """The one prompt format all datasets share (ValueError if they differ) and every task they hold."""
    metas = [json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}
             for path in data_dirs]
    prompt_formats = {meta.get("prompt_format", 1) for meta in metas}
    if len(prompt_formats) != 1:
        raise ValueError(f"all datasets must use one prompt format, got {sorted(prompt_formats)}")
    tasks = sorted({task for meta in metas for task in meta.get("tasks", ["ask", "react", "refuse"])})
    return prompt_formats.pop(), tasks


def load_init(init: Path, device: torch.device, prompt_format: int,
              tasks: list[str]) -> tuple[GPT, Tokenizer, dict[str, Any]]:
    """The pre-trained model, its tokenizer and checkpoint; the vocabulary grows for tokens added after pre-training
    when the data uses them (parse, or the B-plan formats 4-5 with <|open|> and <|imagine|>)."""
    state = torch.load(init, map_location=device)
    model = GPT(GPTConfig(**state["config"])).to(device)
    model.load_state_dict(state["model"])
    tok = Tokenizer(TOKENIZER_DIR / f"{state['tokenizer']}.model")
    uses_extra = "parse" in tasks or prompt_format >= 4
    if uses_extra and model.cfg.vocab_size < tok.vocab_size:
        model = grow_vocab(model, tok.vocab_size)
        print(f"vocabulary grown to {model.cfg.vocab_size} for {tok.vocab_size - tok.sp_size} added tokens", flush=True)
    return model, tok, state


def load_examples(tok: Tokenizer, data_dirs: list[Path], max_len: int) -> tuple[list[Encoded], list[Encoded]]:
    """Every training example and up to VAL_LIMIT validation examples per dataset, encoded."""
    t = time.time()
    train, skipped = [], 0
    val = []
    for path in data_dirs:
        rows, missed = encode_examples(tok, path / "train.jsonl", max_len, limit=None)
        train.extend(rows)
        skipped += missed
        rows, _ = encode_examples(tok, path / "val.jsonl", max_len, limit=VAL_LIMIT)
        val.extend(rows)
    lengths = sorted(len(ids) for ids, _ in train)
    print(f"train {len(train):,} examples (skipped {skipped} > {max_len} tokens), "
          f"length median {lengths[len(lengths) // 2]} max {lengths[-1]} ({time.time() - t:.0f}s)", flush=True)
    return train, val


def lr_at(step: int, steps: int, start: int, peak: float, rewarm: bool) -> float:
    """Linear warm-up over WARMUP_STEPS (again after a resume without optimizer state: Adam's moments start from
    zero), then cosine decay from `peak` to MIN_LR_SHARE of it at the last step."""
    warm = min(1.0, (step - start) / WARMUP_STEPS) if rewarm else 1.0
    return peak * warm * (MIN_LR_SHARE + (1 - MIN_LR_SHARE) * 0.5 * (1 + math.cos(math.pi * step / steps)))


def _data_ref(path: Path) -> str:
    """A dataset path as recorded in the checkpoint: relative to ml/ when inside it."""
    return str(path.resolve().relative_to(ML_DIR)) if path.resolve().is_relative_to(ML_DIR) else str(path)


def _autocast(dtype: str, device: torch.device) -> contextlib.AbstractContextManager[None]:
    """bfloat16 autocast, or nothing for float32."""
    if dtype == "float32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


class _Shuffled:
    """Training examples in random order, epoch after epoch: a new shuffle whenever fewer than a batch are left."""

    def __init__(self, n: int, rng: random.Random) -> None:
        self.n, self.rng = n, rng
        self.order: list[int] = []

    def take(self, k: int) -> list[int]:
        """The next k example indices."""
        if len(self.order) < k:
            self.order = list(range(self.n))
            self.rng.shuffle(self.order)
        return [self.order.pop() for _ in range(k)]


def _free_mps(device: torch.device) -> None:
    """Batches of many lengths: MPS keeps every size it allocated unless told to let go (2026-10-12)."""
    if device.type == "mps":
        torch.mps.empty_cache()


class _Clock:
    """Step times for the progress line, the slowdown watchdog and the --max-minutes limit."""

    def __init__(self, start: int, steps: int, max_minutes: float) -> None:
        self.start, self.steps, self.max_minutes = start, steps, max_minutes
        self.t0 = time.time()
        self.times: list[float] = []

    def stepped(self, step: int, seconds: float, loss: torch.Tensor, lr: float, device: torch.device) -> None:
        """Record one step; stop if training slowed down, and print a progress line every LOG_EVERY steps."""
        self.times.append(seconds)
        if step % WATCH_WINDOW == 0 and slowed_down(self.times):
            raise RuntimeError(f"training slowed down at step {step}; stopped before wasting the GPU")
        if step % LOG_EVERY == 0:
            per_step = (time.time() - self.t0) / (step - self.start)
            recent = sum(self.times[-LOG_EVERY:]) / len(self.times[-LOG_EVERY:])
            print(f"step {step}/{self.steps} loss {loss.item():.4f} lr {lr:.2e} "
                  f"({per_step:.2f}s/step, last 50 {recent:.2f}s)", flush=True)
            _free_mps(device)

    def out_of_time(self) -> bool:
        """Past --max-minutes since the loop started (never with 0)."""
        return bool(self.max_minutes) and time.time() - self.t0 > 60 * self.max_minutes


@dataclass
class _Run:
    """One fine-tuning run: the model and data, and saving to run_dir."""
    args: argparse.Namespace
    model: GPT
    tok: Tokenizer
    state: dict[str, Any]
    device: torch.device
    train: list[Encoded]
    val: list[Encoded]
    tasks: list[str]

    def step(self, optimizer: torch.optim.Optimizer, batch: list[Encoded], lr: float) -> torch.Tensor:
        """One optimizer step at learning rate lr; returns the batch loss."""
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y, m = collate(batch, self.tok.pad_id, self.device, per_example=self.args.loss == "per-example")
        with _autocast(self.args.dtype, self.device):
            loss = self.model(x, y, m)[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss

    def save(self, optimizer: torch.optim.Optimizer, step: int, lr: float, best: float) -> float:
        """Validation loss into metrics.jsonl; last.pt always (with the optimizer), best.pt when it improved.
        Returns the best validation loss so far."""
        vl = val_loss(self.model, self.val, self.tok.pad_id, self.device, batch=VAL_BATCH, n_batches=VAL_BATCHES)
        record = {"step": step, "val_loss": vl, "lr": lr}
        with open(self.args.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        ckpt = {"model": self.model.state_dict(), "config": self.model.cfg.to_dict(),
                "tokenizer": self.state["tokenizer"], "step": step, "val_loss": vl,
                "prompt_format": prompt_format_of(self.args.data),
                "data": [_data_ref(p) for p in [self.args.data, *self.args.extra_data]], "tasks": self.tasks,
                "loss": self.args.loss}
        if vl < best:
            best = vl
            torch.save(ckpt, self.args.run_dir / "best.pt")
        torch.save({**ckpt, "optimizer": optimizer.state_dict()}, self.args.run_dir / "last.pt")
        _free_mps(self.device)
        return best

    def fit(self) -> None:
        """The training loop: shuffled batches, warm-up and cosine schedule, logging, the slowdown watchdog,
        evaluation and saving every --eval-every steps, at the end and at the time limit."""
        args = self.args
        optimizer = make_optimizer(self.model, args.lr, weight_decay=0.0)
        resumed_optimizer = args.start_step > 0 and "optimizer" in self.state
        if resumed_optimizer:
            optimizer.load_state_dict(self.state["optimizer"])
        steps = int(args.epochs * len(self.train) / args.batch)
        start = args.start_step
        shuffled = _Shuffled(len(self.train), random.Random(args.seed + start))
        best = self.state.get("val_loss", float("inf")) if start else float("inf")
        clock = _Clock(start, steps, args.max_minutes)
        for step in range(start + 1, steps + 1):
            t_step = time.time()
            lr = lr_at(step, steps, start, args.lr, rewarm=not resumed_optimizer)
            loss = self.step(optimizer, [self.train[i] for i in shuffled.take(args.batch)], lr)
            clock.stepped(step, time.time() - t_step, loss, lr, self.device)
            out_of_time = clock.out_of_time()
            if step % args.eval_every == 0 or step == steps or out_of_time:
                best = self.save(optimizer, step, lr, best)
            if out_of_time:
                print(f"time limit of {args.max_minutes} minutes reached at step {step}; saved and stopped", flush=True)
                break


def final_eval(run_dir: Path, data: Path, tok: Tokenizer, device: torch.device) -> None:
    """Task metrics of best.pt on the held-out splits (evaluate.py), for engine-made datasets that have them."""
    best_model, _ = ev.load_model(run_dir / "best.pt", device)
    report = {split: ev.run(best_model, tok, split, FINAL_EVAL_N, data_dir=data)["report"]
              for split in ("test_seen", "test_heldout")}
    (run_dir / "eval.json").write_text(json.dumps(report, indent=2))
    print("EVAL", json.dumps(report), flush=True)


def main() -> None:
    """Fine-tune --init on the datasets and save metrics.jsonl, best.pt and last.pt in --run-dir."""
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()
    data_dirs = [args.data, *args.extra_data]
    prompt_format, tasks = dataset_format(data_dirs)
    model, tok, state = load_init(args.init, device, prompt_format, tasks)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.data / "scenes.jsonl").exists() and not is_current(args.data):    # engine-made datasets only
        print(f"WARNING: {args.data} was written by older code; the model will learn the old rules", flush=True)
    train, val = load_examples(tok, data_dirs, args.max_len)
    _Run(args, model, tok, state, device, train, val, tasks).fit()
    if (args.data / "test_seen.jsonl").exists():      # B-plan sales have train/val only (buyers/sales_sft.py)
        final_eval(args.run_dir, args.data, tok, device)


if __name__ == "__main__":
    main()
