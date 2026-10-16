"""Brush strokes as tokens, for a GPT that paints a cat one stroke at a time.

A painting is BOS, then for each stage of catbrush.STAGES a STAGE token followed by that stage's strokes, then EOS.
A stroke is five tokens:

    X       centre x, 1-pixel bins on the 256 canvas
    Y       centre y, 1-pixel bins
    ANGLE   direction in [0, π): the brush is symmetric end to end
    SIZE    length and width, each within catbrush.SIZE_RANGE × the stage's brush, as one token
    PAINT   colour (LEVELS_RGB steps a channel) and opacity (LEVELS_ALPHA steps over catbrush.ALPHA_RANGE)

The stage token carries the brush size, so a stroke never spells out its absolute size. Measured on one AFHQ cat
(2026-10-09): 1,208 strokes are 6,047 tokens; painting with each stage snapped to these bins (`snapper`) ends at an
L1 of 0.0256 to the photo, against 0.0229 unquantised.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import re

import numpy as np
import torch

from . import catbrush as B

BOS, EOS, STAGE0 = 0, 1, 2
NXY, NANGLE, LEVELS_SIZE, LEVELS_RGB, LEVELS_ALPHA = 256, 32, 6, 12, 4
TOKENS_PER_STROKE = 5
FIELDS = {"x": NXY, "y": NXY, "angle": NANGLE, "size": LEVELS_SIZE ** 2, "paint": LEVELS_RGB ** 3 * LEVELS_ALPHA}


def _offsets() -> dict[str, int]:
    out, start = {}, STAGE0 + len(B.STAGES)
    for name, n in FIELDS.items():
        out[name] = start
        start += n
    out["vocab"] = start
    return out


OFFSET = _offsets()
VOCAB = OFFSET["vocab"]


def _bin(x: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
    return np.clip(((np.asarray(x) - lo) / (hi - lo) * n).astype(int), 0, n - 1)


def _centre(i: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
    return lo + (np.asarray(i) + 0.5) * (hi - lo) / n


def stroke_tokens(st: B.Stage, k: int) -> np.ndarray:
    """(n, TOKENS_PER_STROKE) token ids of stage k's strokes."""
    base = np.array(B.STAGES[k][:2], float)
    scale = st["length_width"] / base
    rgb = _bin(st["colour"], 0, 1, LEVELS_RGB)
    paint = ((rgb[:, 0] * LEVELS_RGB + rgb[:, 1]) * LEVELS_RGB + rgb[:, 2]) * LEVELS_ALPHA + \
        _bin(st["alpha"], *B.ALPHA_RANGE, LEVELS_ALPHA)
    return np.stack([OFFSET["x"] + _bin(st["xy"][:, 0], 0, B.SIDE, NXY),
                     OFFSET["y"] + _bin(st["xy"][:, 1], 0, B.SIDE, NXY),
                     OFFSET["angle"] + _bin(np.mod(st["angle"], math.pi), 0, math.pi, NANGLE),
                     OFFSET["size"] + _bin(scale[:, 0], *B.SIZE_RANGE, LEVELS_SIZE) * LEVELS_SIZE
                     + _bin(scale[:, 1], *B.SIZE_RANGE, LEVELS_SIZE),
                     OFFSET["paint"] + paint], 1)


def stroke_params(rows: np.ndarray, k: int) -> B.Stage:
    """Stage k's stroke parameters at the centres of the bins in rows (n, TOKENS_PER_STROKE)."""
    base = np.array(B.STAGES[k][:2], float)
    size, paint = rows[:, 3] - OFFSET["size"], rows[:, 4] - OFFSET["paint"]
    rgb_i = paint // LEVELS_ALPHA
    rgb = np.stack([rgb_i // LEVELS_RGB ** 2, rgb_i // LEVELS_RGB % LEVELS_RGB, rgb_i % LEVELS_RGB], 1)
    return {"xy": np.stack([_centre(rows[:, 0] - OFFSET["x"], 0, B.SIDE, NXY),
                            _centre(rows[:, 1] - OFFSET["y"], 0, B.SIDE, NXY)], 1),
            "angle": _centre(rows[:, 2] - OFFSET["angle"], 0, math.pi, NANGLE),
            "length_width": np.stack([_centre(size // LEVELS_SIZE, *B.SIZE_RANGE, LEVELS_SIZE),
                                      _centre(size % LEVELS_SIZE, *B.SIZE_RANGE, LEVELS_SIZE)], 1) * base,
            "colour": _centre(rgb, 0, 1, LEVELS_RGB),
            "alpha": _centre(paint % LEVELS_ALPHA, *B.ALPHA_RANGE, LEVELS_ALPHA)}


def encode(stages: list[B.Stage]) -> list[int]:
    out = [BOS]
    for k, st in enumerate(stages):
        out.append(STAGE0 + k)
        out += stroke_tokens(st, k).reshape(-1).tolist()
    return out + [EOS]


def decode(tokens: list[int]) -> list[B.Stage]:
    """Tokens → stages of stroke parameters at bin centres. Expects encode's grammar: each stage's strokes are the
    whole TOKENS_PER_STROKE groups between its STAGE token and the next STAGE or EOS."""
    t = np.asarray(tokens)
    marks = np.flatnonzero((t >= STAGE0) & (t < STAGE0 + len(B.STAGES)))
    eos = np.flatnonzero(t == EOS)
    ends = np.r_[marks[1:], eos[0] if len(eos) else len(t)]
    stages = []
    for start, end in zip(marks, ends, strict=True):
        body = t[start + 1:end]
        body = body[:len(body) - len(body) % TOKENS_PER_STROKE]
        stages.append(stroke_params(body.reshape(-1, TOKENS_PER_STROKE), int(t[start]) - STAGE0))
    return stages


def snapper() -> Callable[[B.Stage, int], B.Stage]:
    """For catbrush.PaintSettings(snap=...): a finished stage's parameters moved to the centres of their bins."""
    def snap(stage: B.Stage, k: int) -> B.Stage:
        return stroke_params(stroke_tokens(stage, k), k)
    return snap


def render(stages: list[B.Stage], dev: str = "cpu") -> np.ndarray:
    """Paint stages on white paper with catbrush's renderer, (SIDE, SIDE, 3) in 0–1."""
    ys, xs = torch.meshgrid(torch.arange(B.SIDE) + 0.5, torch.arange(B.SIDE) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1).to(dev)
    canvas = torch.ones(B.SIDE, B.SIDE, 3, device=dev)
    with torch.no_grad():
        for k, st in enumerate(stages):
            canvas = B.composite(canvas, B.Strokes.from_stage(st, B.STAGES[k][:2]).to(dev), grid)
    return canvas.cpu().numpy()


def load_npz(path: str) -> list[B.Stage]:
    """Stages saved by `python -m sellanything_ml.sketch.catbrush` (keys s<stage>_<field>; other keys ignored)."""
    z = np.load(path)
    keys = [key for key in z.files if re.fullmatch(r"s\d+_\w+", key)]
    n = 1 + max(int(key.split("_")[0][1:]) for key in keys)
    return [{key.split("_", 1)[1]: z[key] for key in keys if key.startswith(f"s{i}_")} for i in range(n)]
