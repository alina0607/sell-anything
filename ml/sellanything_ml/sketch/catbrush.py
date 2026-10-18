"""A cat photo as an ordered list of brush strokes, painted big to small like an oil painting.

A stroke is a rounded rectangle with a one-pixel edge and faint bristle streaks along it: centre, length, width,
angle, colour and opacity. Strokes are laid one over another (later strokes on top) on white paper.

The decomposition is fitted stage by stage, coarse to fine: each stage places its strokes where the canvas is still
furthest from the photo, starts them along the photo's edges (the local image structure, which need not follow the
fur) and in the photo's colour there, then adjusts every parameter of that stage's strokes by gradient descent on
the L1 distance to the photo; earlier stages stay fixed.
This is direct optimisation through a differentiable renderer, as in Zou et al., "Stylized Neural Painting"
(CVPR 2021), with Paint Transformer's (Liu et al., ICCV 2021) straight oil stroke.

The owner chose this brush from a comparison of five on 2026-10-09 (docs/devlog/2026-10-09.md); the code of that
comparison, curved, textured and watercolour brushes included, is in the private Kaggle dataset
alinachuang/cat-brush-proto.

    python -m sellanything_ml.sketch.catbrush <photo.png> <mask.png> <out.png>
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

SIDE = 256
# (length, width) in pixels on the SIDE canvas, and strokes per stage; big to small
STAGES = [(96, 40, 24), (56, 22, 64), (32, 12, 160), (18, 7, 360), (10, 4, 600)]
ITERS = 150
SIZE_RANGE = (0.5, 1.5)          # a stroke's length and width, as multiples of its stage's
ALPHA_RANGE = (0.3, 1.0)
EDGE_PX = 0.5                    # edge softness in pixels: coverage falls from 88% to 12% over about 2 px
WINDOW_MARGIN = 4                # coverage 4 px outside the edge is sigmoid(-8) ≈ 3e-4: a window may stop there
# Windows wider than this share of the canvas cost more than painting the whole canvas (measured on MPS:
# 101 px windows 81 ms vs 63 ms dense; 61 px windows 14 ms vs 152 ms)
DENSE_ABOVE = 1 / 3

Stage = dict[str, np.ndarray]    # xy (n, 2) and length_width (n, 2) in px, angle (n,), colour (n, 3), alpha (n,)


class StrokeParams(NamedTuple):
    xy: torch.Tensor             # (n, 2) centres in pixels
    lw: torch.Tensor             # (n, 2) length and width in pixels
    angle: torch.Tensor          # (n,) radians
    colour: torch.Tensor         # (n, 3) RGB in 0–1
    alpha: torch.Tensor          # (n,) opacity


class Brush(NamedTuple):
    """A stroke's geometry, shaped to broadcast against a tensor of pixel positions."""

    x0: torch.Tensor
    y0: torch.Tensor
    half_length: torch.Tensor
    half_width: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor

    @classmethod
    def of(cls, p: StrokeParams, extra_dims: int) -> Brush:
        shape = (-1,) + (1,) * extra_dims
        return cls(p.xy[:, 0].view(shape), p.xy[:, 1].view(shape), (p.lw[:, 0] / 2).view(shape),
                   (p.lw[:, 1] / 2).view(shape), p.angle.cos().view(shape), p.angle.sin().view(shape))


def _logit(p: np.ndarray | torch.Tensor) -> torch.Tensor:
    p = torch.as_tensor(p, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


@dataclass
class Strokes:
    """Unconstrained parameters of one stage's strokes; `params` maps them to pixels, radians and colours."""

    xy: torch.Tensor             # (n, 2) logits of the centre in [0, SIDE)
    size: torch.Tensor           # (n, 2) logits of length and width within SIZE_RANGE × base
    angle: torch.Tensor          # (n,) radians
    colour: torch.Tensor         # (n, 3) logits of RGB
    alpha: torch.Tensor          # (n,) logits of opacity within ALPHA_RANGE
    base: tuple[float, float]    # the stage's (length, width)

    def tensors(self) -> list[torch.Tensor]:
        return [self.xy, self.size, self.angle, self.colour, self.alpha]

    def params(self) -> StrokeParams:
        lo, hi = SIZE_RANGE
        a_lo, a_hi = ALPHA_RANGE
        xy = torch.sigmoid(self.xy) * SIDE
        lw = (lo + (hi - lo) * torch.sigmoid(self.size)) * torch.tensor(self.base, device=xy.device)
        return StrokeParams(xy, lw, self.angle, torch.sigmoid(self.colour),
                            a_lo + (a_hi - a_lo) * torch.sigmoid(self.alpha))

    @classmethod
    def from_stage(cls, st: Stage, base: tuple[float, float]) -> Strokes:
        """The inverse of `params`: strokes whose parameters are the given values (e.g. snapped to token bins)."""
        lo, hi = SIZE_RANGE
        a_lo, a_hi = ALPHA_RANGE
        return cls(xy=_logit(st["xy"] / SIDE), size=_logit((st["length_width"] / np.array(base) - lo) / (hi - lo)),
                   angle=torch.as_tensor(st["angle"], dtype=torch.float32), colour=_logit(st["colour"]),
                   alpha=_logit((st["alpha"] - a_lo) / (a_hi - a_lo)), base=tuple(base))

    def to(self, dev: str | torch.device) -> Strokes:
        for t in self.tensors():
            t.data = t.data.to(dev)
        return self


def rounded_rect(brush: Brush, px: torch.Tensor, py: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Coverage and bristle shading at pixel centres (px, py), from the signed distance to a rounded rectangle in
    pixels (Quilez's box SDF; corner radius half the half-width), so the edge is equally sharp for a 4 px and a
    40 px stroke."""
    dx, dy = px - brush.x0, py - brush.y0
    u, v = dx * brush.cos + dy * brush.sin, -dx * brush.sin + dy * brush.cos
    rho = 0.5 * brush.half_width
    qx, qy = u.abs() - (brush.half_length - rho).clamp(min=0), v.abs() - (brush.half_width - rho)
    outside = (qx.clamp(min=0) ** 2 + qy.clamp(min=0) ** 2 + 1e-8).sqrt()   # eps: sqrt has no slope at 0
    sdf = outside + torch.maximum(qx, qy).clamp(max=0) - rho
    return torch.sigmoid(-sdf / EDGE_PX), 1 + 0.06 * torch.sin(v / brush.half_width * 9.0 + brush.x0)


def window_half(base: tuple[float, float]) -> int:
    """Half the side of the square that holds a stroke of the stage's largest size, edge included."""
    hi = SIZE_RANGE[1] / 2
    return int(math.ceil(math.hypot(hi * base[0], hi * base[1]) + WINDOW_MARGIN))


def composite_dense(canvas: torch.Tensor, p: StrokeParams, grid: torch.Tensor) -> torch.Tensor:
    """Every stroke at every pixel: out = Σ_i a_i c_i Π_{j>i}(1 − a_j) + canvas Π_j (1 − a_j)."""
    cover, shade = rounded_rect(Brush.of(p, 2), grid[None, ..., 0], grid[None, ..., 1])
    a = cover * p.alpha[:, None, None]
    keep = torch.flip(torch.cumprod(torch.flip(1 - a, [0]), 0), [0])    # Π_{j>=i}(1 − a_j)
    after = torch.cat([keep[1:], torch.ones_like(keep[:1])])            # Π_{j>i}
    paint = (a * after)[..., None] * (p.colour[:, None, None, :] * shade[..., None]).clamp(0, 1)
    return paint.sum(0) + canvas * keep[0][..., None]


def composite_windowed(canvas: torch.Tensor, p: StrokeParams, half: int) -> torch.Tensor:
    """The same sum, each stroke evaluated only in its own (2·half + 1)² window.

    The windows' entries are grouped by pixel (sorted by pixel, then stroke) and laid into a (pixels touched × most
    strokes on one pixel) matrix of log(1 − a); a reversed cumulative sum along each row gives every entry's
    Π_{j>i}(1 − a_j) in one step."""
    h, w = canvas.shape[:2]
    n, dev = len(p.xy), p.xy.device
    off = torch.arange(-half, half + 1, device=dev)
    oy, ox = torch.meshgrid(off, off, indexing="ij")
    ix = p.xy[:, 0].detach().floor().long()[:, None] + ox.reshape(1, -1)           # (n, window pixels)
    iy = p.xy[:, 1].detach().floor().long()[:, None] + oy.reshape(1, -1)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    cover, shade = rounded_rect(Brush.of(p, 1), ix + 0.5, iy + 0.5)
    a = (cover * p.alpha[:, None] * valid).clamp(max=1 - 1e-6)
    col = (p.colour[:, None, :] * shade[..., None]).clamp(0, 1)
    pix = (iy.clamp(0, h - 1) * w + ix.clamp(0, w - 1)).reshape(-1)
    order = torch.argsort(pix * n + torch.arange(n, device=dev).repeat_interleave(a.shape[1]))
    px_s, a_s, col_s = pix[order], a.reshape(-1)[order], col.reshape(-1, 3)[order]
    first = torch.ones_like(px_s, dtype=torch.bool)
    first[1:] = px_s[1:] != px_s[:-1]
    seg = torch.cumsum(first.long(), 0) - 1
    starts = torch.nonzero(first).squeeze(1)
    rank = torch.arange(len(px_s), device=dev) - starts[seg]
    log_keep = torch.zeros(len(starts), int(rank.max()) + 1, device=dev).index_put((seg, rank), torch.log1p(-a_s))
    later = torch.flip(torch.cumsum(torch.flip(log_keep, [1]), 1), [1]) - log_keep  # Σ_{j>i} log(1 − a_j)
    paint = torch.zeros(h * w, 3, device=dev).index_add(0, px_s, (a_s * torch.exp(later[seg, rank]))[:, None] * col_s)
    keep = torch.zeros(h * w, device=dev).index_add(0, px_s[starts], log_keep.sum(1))
    return (canvas.reshape(-1, 3) * torch.exp(keep)[:, None] + paint).reshape(h, w, 3)


def composite(canvas: torch.Tensor, strokes: Strokes, grid: torch.Tensor) -> torch.Tensor:
    """Paint one stage's strokes over the canvas, by windows where they are small and densely where they are not."""
    half = window_half(strokes.base)
    if 2 * half + 1 > DENSE_ABOVE * SIDE:
        return composite_dense(canvas, strokes.params(), grid)
    return composite_windowed(canvas, strokes.params(), half)


def orientation(target: torch.Tensor) -> torch.Tensor:
    """Per-pixel angle along the image's local structure (perpendicular to the smoothed gradient), (H, W)."""
    g = target.mean(-1)[None, None]
    kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=g.dtype) / 8
    gx, gy = F.conv2d(g, kx, padding=1), F.conv2d(g, kx.transpose(2, 3), padding=1)
    blur = lambda t: F.avg_pool2d(t, 9, 1, 4)                           # noqa: E731
    jxx, jyy, jxy = blur(gx * gx), blur(gy * gy), blur(gx * gy)
    return (0.5 * torch.atan2(2 * jxy, jxx - jyy) + math.pi / 2)[0, 0]


@dataclass
class Photo:
    """What a painting is fitted to: the photo on white paper, where the cat is, and the direction along its edges."""

    target: torch.Tensor         # (SIDE, SIDE, 3) in 0–1
    mask: torch.Tensor           # (SIDE, SIDE), 1 on the cat
    angles: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.angles = orientation(self.target)


@dataclass
class PaintSettings:
    seed: int = 0
    dev: str = "cpu"
    iters: int = ITERS
    snap: Callable[[Stage, int], Stage] | None = None    # e.g. brushtokens.snapper(): finished stages onto the token grid
    log: Callable[[str], None] = print
    stages: int = len(STAGES)                             # paint only the first this many stages


def new_stage(photo: Photo, canvas: torch.Tensor, k: int, rng: np.random.Generator) -> Strokes:
    """Stage k's strokes, placed with probability ∝ the canvas's error, along the edges, in the photo's colour."""
    length, width, n = STAGES[k]
    err = ((photo.target - canvas).abs().sum(-1) * photo.mask).flatten().numpy() + 1e-6
    idx = rng.choice(err.size, n, replace=False, p=err / err.sum())
    ys, xs = idx // SIDE, idx % SIDE
    xy = torch.tensor(np.c_[xs, ys] + 0.5, dtype=torch.float32)
    return Strokes(xy=_logit(xy / SIDE), size=torch.zeros(n, 2), angle=photo.angles.flatten()[idx].clone(),
                   colour=_logit(photo.target[ys, xs]), alpha=torch.full((n,), 1.5), base=(length, width))


def paint(photo: Photo, settings: PaintSettings | None = None) -> tuple[list[Stage], list[np.ndarray]]:
    """The stages' stroke parameters and the canvas after each stage.

    With settings.snap, each finished stage's parameters move onto the token grid before the next stage starts, so
    later strokes paint over the snapping error, and the result is exactly what the tokens say."""
    cfg = settings or PaintSettings()
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    ys, xs = torch.meshgrid(torch.arange(SIDE) + 0.5, torch.arange(SIDE) + 0.5, indexing="ij")
    grid = torch.stack([xs, ys], -1).to(cfg.dev)
    target = photo.target.to(cfg.dev)
    canvas = torch.ones_like(target)
    stages, frames = [], []
    for k, (length, width, n) in enumerate(STAGES[:cfg.stages]):
        s = new_stage(photo, canvas.cpu(), k, rng).to(cfg.dev)
        for t in s.tensors():
            t.requires_grad_(True)
        opt = torch.optim.Adam(s.tensors(), lr=0.05)
        for _ in range(cfg.iters):
            loss = (composite(canvas, s, grid) - target).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            p = s.params()
            st = {"xy": p.xy, "length_width": p.lw, "angle": p.angle, "colour": p.colour, "alpha": p.alpha}
            st = {key: v.detach().cpu().numpy() for key, v in st.items()}
            if cfg.snap is not None:
                st = cfg.snap(st, k)
                s = Strokes.from_stage(st, (length, width)).to(cfg.dev)
            canvas = composite(canvas, s, grid)
        stages.append(st)
        frames.append(canvas.cpu().numpy())
        cfg.log(f"  stage {k + 1}: {n} strokes of {length}×{width}, L1 {(canvas - target).abs().mean().item():.4f}")
    return stages, frames


def load(photo_path: str, mask_path: str, box: tuple[int, int, int, int] | None = None) -> Photo:
    """A photo (cut to `box`, (left, top, right, bottom) in its pixels, if given) resized to SIDE, with everything
    outside the subject mask turned to white paper."""
    from PIL import Image
    photo, mask = Image.open(photo_path).convert("RGB"), Image.open(mask_path).convert("L")
    if box is not None:
        mask = mask.resize(photo.size, Image.BILINEAR).crop(box)
        photo = photo.crop(box)
    img = np.asarray(photo.resize((SIDE, SIDE), Image.LANCZOS), np.float32) / 255
    m = np.asarray(mask.resize((SIDE, SIDE), Image.BILINEAR), np.float32) / 255
    target = img * m[..., None] + (1 - m[..., None])
    return Photo(torch.from_numpy(target), torch.from_numpy(m))


if __name__ == "__main__":
    from PIL import Image
    photo_path, mask_path, out = sys.argv[1:4]
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    photo = load(photo_path, mask_path)
    t0 = time.time()
    stages, frames = paint(photo, PaintSettings(dev=dev))
    row = np.concatenate([photo.target.numpy()] + frames, 1)
    Image.fromarray((row.clip(0, 1) * 255).astype(np.uint8)).save(out)
    np.savez(out.rsplit(".", 1)[0] + ".npz", **{f"s{i}_{k}": v for i, st in enumerate(stages) for k, v in st.items()})
    print(f"{sum(len(s['alpha']) for s in stages)} strokes, {time.time() - t0:.0f} s on {dev}")
