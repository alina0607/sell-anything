"""Brush GPT: a decoder-only transformer that paints a cat's face one brush stroke at a time.

Data: AFHQ cat faces painted with the straight brush by tools/paint_afhq.py (catbrush, the first four stages:
24, 64, 160 and 360 strokes, each stage snapped to the token grid before the next is painted). Every painting is
therefore exactly brushtokens.encode's 3,046 tokens:

    BOS  STAGE0 (x y angle size paint) × 24  STAGE1 … × 64  STAGE2 … × 160  STAGE3 … × 360  EOS

Every painting has the same layout, so each position's slot (which stage, which of the five fields) is fixed and
known; `layout` gives the token range allowed at each position, and the model gets the slot directly from a
learned position embedding (abs_pos), as Line GPT needed.

Within a stage the strokes are in the order the painter placed them (drawn at random, weighted by how far the
canvas still was from the photo); later strokes cover earlier ones, so the order is part of the picture and is
kept as it is.

Mirror images are made in token space: x bin i → 255 − i and angle bin j → 31 − j (1-pixel x bins on the
256-pixel canvas and 32 angle bins over [0, π) are symmetric). The renderer's faint bristle streaks depend on a
stroke's x, so a mirrored painting renders very slightly differently from the mirrored picture; `data` measures
how much.

Stroke order inside a stage (`order`): "list" is the painter's order, which is random (it fits a stage's strokes
together, from starting points drawn at random), so where the next stroke goes cannot be predicted (the trial: x
loss 5.53 against ln 256 = 5.55). "overlap" keeps the order of every two strokes that overlap (there the later one is
on top, so the order is part of the picture) and puts the rest top to bottom in 32-pixel bands, left to right: the
same picture to float precision, and the next stroke is 29 px away on average in the last stage instead of 115
(devlog 2026-10-11 §2). Mirror images are re-ordered after mirroring, so both follow the same rule.

One output layer per field (`heads`): every position is a known field (`layout`), so the token there is chosen only
among that field's tokens (x among the 256 x bins, colour among the 6,912 colours, …), in training and in sampling.
It is done by using, at each position, only that field's rows of the shared (tied) output table: the same as five
output layers, one per field, that share their weights with the input embedding. The model still decides every value;
what is fixed is the data's own layout (owner, 2026-10-11). Without it, the trial's model put 15% of sampled tokens in
the wrong field and 0 of 256 paintings decoded.

Face points first (`points`, devlog 2026-10-11 §8): the painting starts with the cat's nine face points (eyes,
mouth, ears; the landmark model's for AFHQ, hand-labelled for the CAT dataset) as x and y tokens, then the strokes.
The model generates the points itself and paints after them, so it decides where the eyes are before the small
strokes that make them. Eye weighting (`eye_weight`): the loss of every token of a stroke whose centre lies within
EYE_ZONE eye-to-eye distances of an eye counts eye_weight times (1 = off). The validation loss is always reported
unweighted, with the eye strokes' own loss ("eye") next to it.

Aligned faces (`align` stage, devlog 2026-10-11 §9): the shape GPT, whose cats always had eyes, drew catface.py's
faces aligned by their points (eyes level, at fixed pixels). The same is done to finished paintings without
painting again: every stroke is moved, turned and scaled so the eyes lie level, their midpoint at (SIDE/2, the
mean height) and their distance the median one; the size tokens are binned again and clipped to their range, and
the stroke order is kept. Written to a separate file (ALIGNED_FILE) with the moved face points.

Splits: AFHQ's own test folder is the test set (not painted yet in the first trial); of AFHQ's train cats, a
stable 8% chosen by a hash of the file name is validation, so painting more cats never moves a cat between splits.

    python -m sellanything_ml.sketch.brushgpt data                       # paintings → tokens.npz, checks
    python -m sellanything_ml.sketch.brushgpt align                      # tokens.npz → tokens_aligned.npz
    python -m sellanything_ml.sketch.brushgpt data --paint-dirs <afhq> <cat> --data <file>   # several folders
    python -m sellanything_ml.sketch.brushgpt align --data <file> --aligned <out> --align-ref tokens_runs_HI.npz
    python -m sellanything_ml.sketch.brushgpt train --name t400 --cats 400
    python -m sellanything_ml.sketch.brushgpt eval --name t400 --n 256
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import brushtokens as T
from . import catbrush as B

if TYPE_CHECKING:
    import torch

    from ..llm.model import GPT

ML_DIR = Path(__file__).resolve().parents[2]
PAINT_DIR = ML_DIR / "data" / "processed" / "afhq_brush_face"
DATA_FILE = ML_DIR / "data" / "processed" / "brushgpt" / "tokens.npz"
ALIGNED_FILE = DATA_FILE.with_name("tokens_aligned.npz")
RUN_DIR = ML_DIR / "runs" / "brushgpt"
STAGES = 4
STROKES = [n for *_, n in B.STAGES[:STAGES]]
SEQ = 2 + STAGES + T.TOKENS_PER_STROKE * sum(STROKES)        # 3,046
VAL_PERCENT = 8
FIELDS = ("x", "y", "angle", "size", "paint")
SLOTS = ("marker",) + FIELDS + ("point",)                      # marker: BOS, a STAGE token or EOS
POINTS = 9                                                    # face points: 2 eyes, mouth, 3 per ear (CAT order)
MIRROR_POINTS = [1, 0, 2, 8, 7, 6, 5, 4, 3]                   # which point each becomes in the mirror image
PREFIX = 2 * POINTS                                           # point tokens after BOS when `points` is on
EYE_ZONE = 0.35                                               # eye-region radius, in eye-to-eye distances


def layout(points: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every position of a painting: the lowest and highest token allowed there, and its slot (index into
    SLOTS). With `points`, the nine face points' x and y tokens come right after BOS."""
    lo, hi, slot = [T.BOS], [T.BOS], [0]
    for _ in range(POINTS if points else 0):
        for f in ("x", "y"):
            lo.append(T.OFFSET[f])
            hi.append(T.OFFSET[f] + T.NXY - 1)
            slot.append(len(SLOTS) - 1)
    for k, n in enumerate(STROKES):
        lo.append(T.STAGE0 + k)
        hi.append(T.STAGE0 + k)
        slot.append(0)
        for _ in range(n):
            for f, name in enumerate(FIELDS):
                lo.append(T.OFFSET[name])
                hi.append(T.OFFSET[name] + T.FIELDS[name] - 1)
                slot.append(f + 1)
    lo.append(T.EOS)
    hi.append(T.EOS)
    slot.append(0)
    return np.array(lo), np.array(hi), np.array(slot)


LO, HI, SLOT = layout()
assert len(LO) == SEQ
LAYOUTS = {False: (LO, HI, SLOT), True: layout(True)}


def layout_of(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The layout token rows follow, told by their length (SEQ, or SEQ + PREFIX with face points)."""
    return LAYOUTS[rows.shape[-1] == SEQ + PREFIX]


def point_tokens(points: np.ndarray) -> np.ndarray:
    """(n, POINTS, 2) canvas pixels → (n, PREFIX) tokens: x and y of each point, binned like a stroke's centre."""
    x = T.OFFSET["x"] + np.clip(np.floor(points[..., 0]), 0, T.NXY - 1).astype(np.int64)
    y = T.OFFSET["y"] + np.clip(np.floor(points[..., 1]), 0, T.NXY - 1).astype(np.int64)
    return np.stack([x, y], -1).reshape(len(points), PREFIX)


def mirror_points(points: np.ndarray) -> np.ndarray:
    """Face points of the mirror images: x → SIDE − x, and left and right points swap."""
    out = points.copy()
    out[..., 0] = B.SIDE - out[..., 0]
    return out[:, MIRROR_POINTS]


def eye_strokes(rows: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(n, len) bool over token positions: the five tokens of every stroke whose centre lies within EYE_ZONE
    eye-to-eye distances of either eye (points 0 and 1)."""
    lo, _, slot = layout_of(rows)
    xs = np.flatnonzero(slot == 1 + FIELDS.index("x"))              # each stroke's first token
    cx = (rows[:, xs] - T.OFFSET["x"] + 0.5) * B.SIDE / T.NXY
    cy = (rows[:, xs + 1] - T.OFFSET["y"] + 0.5) * B.SIDE / T.NXY
    eyes = points[:, :2]                                             # (n, 2, 2)
    r = EYE_ZONE * np.linalg.norm(eyes[:, 0] - eyes[:, 1], axis=1)   # (n,)
    near = np.zeros(cx.shape, bool)
    for e in range(2):
        near |= np.hypot(cx - eyes[:, e, :1], cy - eyes[:, e, 1:]) < r[:, None]
    out = np.zeros(rows.shape, bool)
    for k in range(T.TOKENS_PER_STROKE):
        out[:, xs + k] = near
    return out


def split_of(rel: str) -> str:
    """'test' for AFHQ's test folder; otherwise 'val' for a stable VAL_PERCENT% of cats, else 'train'."""
    if rel.startswith("test/"):
        return "test"
    h = int(hashlib.sha1(Path(rel).stem.encode()).hexdigest(), 16)
    return "val" if h % 100 < VAL_PERCENT else "train"


def mirror(rows: np.ndarray) -> np.ndarray:
    """Left-right mirror images of token rows (n, SEQ) (strokes only; the face points are mirrored by
    mirror_points)."""
    out = rows.copy()
    x, a = SLOT == 1 + FIELDS.index("x"), SLOT == 1 + FIELDS.index("angle")
    out[:, x] = 2 * T.OFFSET["x"] + T.NXY - 1 - rows[:, x]
    out[:, a] = 2 * T.OFFSET["angle"] + T.NANGLE - 1 - rows[:, a]
    return out


def valid(rows: np.ndarray) -> np.ndarray:
    """(n, len) bool: is each token inside the range its position allows?"""
    lo, hi, _ = layout_of(rows)
    return (rows >= lo) & (rows <= hi)


def strokes_only(rows: np.ndarray) -> np.ndarray:
    """Token rows without their face points (unchanged if they have none)."""
    return np.delete(rows, np.s_[1:1 + PREFIX], axis=-1) if rows.shape[-1] == SEQ + PREFIX else rows


def render(row: np.ndarray, dev: str = "cpu") -> np.ndarray:
    """A token row painted on white paper, (SIDE, SIDE, 3) in 0–1."""
    return T.render(T.decode(strokes_only(row).tolist()), dev)


BAND = 32                            # px: the height of the rows strokes are sorted in, top to bottom


def overlap_order(st: B.Stage) -> np.ndarray:
    """Stroke indices of a stage in "overlap" order: a stroke that overlaps an earlier one stays after it (overlap:
    centres closer than half the two diagonals plus catbrush's window margin, so never too few); otherwise top to
    bottom in BAND-pixel rows, then left to right."""
    import heapq
    xy, lw = st["xy"], st["length_width"]
    n = len(xy)
    r = 0.5 * np.hypot(lw[:, 0], lw[:, 1]) + B.WINDOW_MARGIN
    dist = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    before = np.triu(dist < r[:, None] + r[None, :], 1)              # before[i, j]: i must come before j
    waiting = before.sum(0)
    key = [(int(xy[i, 1] // BAND), float(xy[i, 0]), i) for i in range(n)]
    ready = [key[i] for i in range(n) if waiting[i] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        i = heapq.heappop(ready)[2]
        order.append(i)
        for j in np.flatnonzero(before[i]):
            waiting[j] -= 1
            if waiting[j] == 0:
                heapq.heappush(ready, key[j])
    return np.array(order)


def reorder(row: np.ndarray) -> np.ndarray:
    """A token row with every stage's strokes in "overlap" order (same picture)."""
    stages = [{f: v[overlap_order(s)] for f, v in s.items()} for s in T.decode(row.tolist())]
    return np.array(T.encode(stages), dtype=row.dtype)


CAT_PAINT_DIR = ML_DIR / "data" / "processed" / "cat_brush_face"
# A CAT-dataset cat is left out when its eyes were not painted well (owner, 2026-10-12: drop those cats, the threshold
# set from the distribution; set 2026-10-16): the landmark model's eye confidence on the painting is below 0.85 of
# that on the photo it copies, or it finds the painting's eyes more than 0.2 eye-to-eye distances from the
# hand-labelled ones. Of the 6,965 painted cats with eye measures: 73 + 73, 11 by both, 135 left out (1.9%).
CAT_EYE_CONF_RATIO, CAT_EYE_ERR = 0.85, 0.2


def _painted(paint_dir: Path) -> dict[str, str]:
    """{.npz stem: cat id} of the paintings in paint_dir that pass the data rules. AFHQ paintings are named by the
    photo's stem (flickr_cat_000002), CAT-dataset ones by the id with / → __ (CAT_DATASET_02__CAT_04__00000909_011),
    since CAT file names repeat between folders."""
    log = [json.loads(line) for name in ("log.jsonl", "log_modal.jsonl") if (paint_dir / name).exists()
           for line in (paint_dir / name).read_text().splitlines()]          # the Mac's log and Modal's (all kept)
    ids = {}
    for e in log:
        if "skipped" in e or not e.get("kept", True):                     # cats failing the data rules stay out
            continue
        if "eye_conf" in e and (e["eye_conf"] < CAT_EYE_CONF_RATIO * e["eye_conf_photo"] or e["eye_err"] > CAT_EYE_ERR):
            continue
        for stem in (Path(e["id"]).stem, Path(e["id"].replace("/", "__")).stem):
            if (paint_dir / f"{stem}.npz").exists():
                ids[stem] = e["id"]
    return ids


def stage_data(paint_dir: Path | list[Path] = PAINT_DIR, out: Path = DATA_FILE) -> dict[str, object]:
    """Every finished painting (of one folder, or several: AFHQ and the CAT dataset) as a token row, with its id and
    split; round-trip and mirror checks."""
    dirs = paint_dir if isinstance(paint_dir, list) else [paint_dir]
    found = [(d, name, cat) for d in dirs for name, cat in sorted(_painted(d).items())]
    names = [name for _, name, _ in found]
    rows = np.empty((len(found), SEQ), np.uint16)
    for i, (d, name, _) in enumerate(found):
        tok = T.encode(T.load_npz(str(d / f"{name}.npz")))
        if len(tok) != SEQ:
            raise ValueError(f"{name}: {len(tok)} tokens, expected {SEQ}")
        rows[i] = tok
    rel = np.array([cat for *_, cat in found])
    split = np.array([split_of(r) for r in rel])
    points = np.concatenate([face_points(d, [n for dd, n, _ in found if dd == d],
                                         np.array([c for dd, _, c in found if dd == d])) for d in dirs])
    if not valid(rows).all():
        raise ValueError("a painting has a token outside its slot's range")
    if not (mirror(mirror(rows)) == rows).all():
        raise ValueError("mirroring twice does not give the painting back")
    # decode → encode gives the same tokens (the paintings were snapped to the bins while they were painted)
    roundtrip = all(T.encode(T.decode(rows[i].tolist())) == rows[i].tolist() for i in range(min(20, len(rows))))
    # how far a mirrored painting's render is from the mirrored render (bristle streaks depend on x)
    a, m = render(rows[0]), render(mirror(rows[:1])[0])
    mirror_l1 = float(np.abs(a[:, ::-1] - m).mean())
    mirrored = mirror(rows)
    ordered = np.stack([reorder(r) for r in rows])
    mirrored_ordered = np.stack([reorder(r) for r in mirrored])
    reorder_change = float(np.abs(render(rows[0]) - render(ordered[0])).max())
    if reorder_change > 1e-4:
        raise ValueError(f"re-ordering changed the picture by {reorder_change}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, tokens=rows, tokens_mirror=mirrored, tokens_overlap=ordered, tokens_mirror_overlap=mirrored_ordered,
             ids=rel, split=split, points=points, points_mirror=mirror_points(points))
    report = {"paintings": len(names), "tokens_each": SEQ, "vocab": T.VOCAB,
              "splits": {s: int((split == s).sum()) for s in ("train", "val", "test")},
              "by_folder": {d.name: {s: int(sum(dd == d and sp == s for (dd, *_), sp in zip(found, split, strict=True)))
                                     for s in ("train", "val", "test")} for d in dirs},
              "roundtrip_first_20": roundtrip, "mirror_render_l1": round(mirror_l1, 5),
              "reorder_max_pixel_change": reorder_change}
    print(json.dumps(report), flush=True)
    return report


def align_rows(rows: np.ndarray, points: np.ndarray, mid: np.ndarray, dist: float) -> tuple[np.ndarray, np.ndarray]:
    """Token rows (strokes only) and their face points moved so the eyes are level, their midpoint at `mid` and
    their distance `dist`: each stroke's centre is moved, its angle turned and its length and width scaled."""
    out, moved = np.empty_like(rows), np.empty_like(points)
    for i, (row, p) in enumerate(zip(rows, points, strict=True)):
        v = p[1] - p[0]
        turn, s = -math.atan2(v[1], v[0]), dist / float(np.hypot(*v))
        a = s * np.array([[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]])
        centre = p[:2].mean(0)
        stages = []
        for st in T.decode(row.tolist()):
            st = dict(st)
            st["xy"] = (st["xy"] - centre) @ a.T + mid
            st["angle"] = np.mod(st["angle"] + turn, math.pi)
            st["length_width"] = st["length_width"] * s
            stages.append(st)
        out[i] = T.encode(stages)
        moved[i] = (p - centre) @ a.T + mid
    return out, moved


def align_data(data: Path = DATA_FILE, out: Path = ALIGNED_FILE, ref: Path | None = None) -> dict[str, object]:
    """Every painting of `data` with its face aligned (see the module docstring), the mirror images too. With `ref`
    (an aligned file), its eye midpoint and distance are used instead of this data's own, so cats that are in both
    files get exactly the same tokens (and losses on them compare between runs)."""
    z = dict(np.load(data))
    train_pts = z["points"][z["split"] == "train"]
    mid = np.array([B.SIDE / 2, float(train_pts[:, :2, 1].mean())])     # x at the centre, so mirrors stay aligned
    dist = float(np.median(np.linalg.norm(train_pts[:, 0] - train_pts[:, 1], axis=1)))
    if ref is not None:
        r = np.load(ref)
        mid, dist = r["align_mid"], float(r["align_dist"])
    for side in ("", "_mirror"):
        pts = z["points" + side]
        for order in ("", "_overlap"):
            z["tokens" + side + order], moved = align_rows(z["tokens" + side + order], pts, mid, dist)
        z["points" + side] = moved
    if not all(valid(z[k]).all() for k in z if k.startswith("tokens")):
        raise ValueError("an aligned painting has a token outside its slot's range")
    np.savez(out, **z, align_mid=mid, align_dist=dist)
    eyes = z["points"][:, :2]
    report = {"paintings": len(z["ids"]), "mid": mid.round(2).tolist(), "eye_distance": round(dist, 2),
              "eye_spread_px": round(float(eyes.std(0).mean()), 3)}
    print(json.dumps(report), flush=True)
    return report


LABELS = ML_DIR / "data" / "processed" / "afhq_cats_mae_pad1.5.npz"     # the landmark model's AFHQ points


def face_points(paint_dir: Path, names: list[str], rel: np.ndarray) -> np.ndarray:
    """(n, POINTS, 2) face points in each painting's canvas pixels: stored in the painting's .npz (CAT dataset), or
    the landmark model's AFHQ labels moved into the painting's face square."""
    labels = np.load(LABELS)
    where = {str(r): i for i, r in enumerate(labels["ids"])}
    out = np.zeros((len(names), POINTS, 2))
    for i, (name, r) in enumerate(zip(names, rel, strict=True)):
        z = np.load(paint_dir / f"{name}.npz")
        if "points" in z.files:
            out[i] = z["points"]
            continue
        box = z["box"]
        out[i] = (labels["points"][where[str(r)]] - box[:2]) / (box[2] - box[0]) * B.SIDE
    return out


@dataclass
class Config:
    name: str = "trial"
    cats: int = 0                    # train cats used (the first of a fixed shuffle); 0 = all of them
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 8
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.1
    batch: int = 8
    accum: int = 2
    max_steps: int = 2000
    warmup: int = 100
    eval_every: int = 100
    patience: int = 6                # stop after this many evaluations without a better validation loss
    stop_after_min: float = 0.0      # wall-clock limit in minutes (0 = none)
    seed: int = 0
    mirror: bool = True
    order: str = "list"              # stroke order inside a stage: "list" (the painter's) or "overlap"
    heads: bool = False              # one output layer per field (see the module docstring)
    points: bool = False             # the face points first (see the module docstring)
    eye_weight: float = 1.0          # loss weight of eye-region strokes (1 = off)


def model_config(cfg: Config):
    from ..llm.model import GPTConfig
    return GPTConfig(vocab_size=T.VOCAB, d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                     n_kv_heads=cfg.n_kv_heads, max_seq_len=SEQ + (PREFIX if cfg.points else 0),
                     dropout=cfg.dropout, abs_pos=True)


def device() -> str:
    """BRUSHGPT_DEVICE if set (e.g. cpu, to keep a test off a GPU another job is using), else the best there is."""
    import os

    import torch
    if os.environ.get("BRUSHGPT_DEVICE"):
        return os.environ["BRUSHGPT_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _kept(z, split: str, cats: int) -> np.ndarray:
    keep = np.flatnonzero(z["split"] == split)
    if split == "train" and cats:
        keep = np.random.default_rng(0).permutation(keep)
        if cats > len(keep):
            raise ValueError(f"asked for {cats} train cats, only {len(keep)} painted")
        keep = np.sort(keep[:cats])
    return keep


def load_rows(data: Path, split: str, cats: int = 0, order: str = "list",
              mirrored: bool = False, points: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Token rows (int64) and ids of a split, in the given stroke order, mirrored or not, with the face points first
    if `points`; for train, the first `cats` of a fixed shuffle (0 = all)."""
    z = np.load(data)
    keep = _kept(z, split, cats)
    key = "tokens" + ("_mirror" if mirrored else "") + ("_overlap" if order == "overlap" else "")
    rows = z[key][keep].astype(np.int64)
    if points:
        pts = z["points_mirror" if mirrored else "points"][keep]
        rows = np.concatenate([rows[:, :1], point_tokens(pts), rows[:, 1:]], 1)
    return rows, z["ids"][keep]


def load_points(data: Path, split: str, cats: int = 0, mirrored: bool = False) -> np.ndarray:
    """The face points (n, POINTS, 2) of the rows load_rows gives for the same arguments."""
    z = np.load(data)
    return z["points_mirror" if mirrored else "points"][_kept(z, split, cats)]


def allowed(dev: str, points: bool = False) -> "torch.Tensor":
    """(len, VOCAB) bool: the tokens each position may hold (its field's tokens)."""
    import torch
    lo, hi, _ = LAYOUTS[points]
    v = torch.arange(T.VOCAB)
    return ((v[None] >= torch.from_numpy(lo)[:, None]) & (v[None] <= torch.from_numpy(hi)[:, None])).to(dev)


def field_logits(logits: "torch.Tensor", start: int, mask: "torch.Tensor | None") -> "torch.Tensor":
    """With one output layer per field: the logits for positions start, start+1, … with every token outside the
    position's field removed. mask is `allowed(dev)` or None (one shared output layer)."""
    if mask is None:
        return logits
    return logits.masked_fill(~mask[start:start + logits.size(1)], float("-inf"))


def slot_losses(model: "GPT", rows: np.ndarray, dev: str, batch: int = 8,
                mask: "torch.Tensor | None" = None, eyes: np.ndarray | None = None) -> dict[str, float]:
    """Mean cross-entropy per token over the rows: overall (strokes and markers only, so runs with and without face
    points compare), for each slot, and over the eye-region strokes if `eyes` (eye_strokes of the rows) is given."""
    import torch
    import torch.nn.functional as F
    slot = torch.from_numpy(layout_of(rows)[2][1:]).to(dev)
    tot = torch.zeros(len(SLOTS), device=dev)
    cnt = torch.zeros(len(SLOTS), device=dev)
    eye_tot, eye_cnt = 0.0, 0.0
    model.eval()
    with torch.no_grad(), autocast(dev):
        for i in range(0, len(rows), batch):
            t = torch.from_numpy(rows[i:i + batch]).to(dev)
            logits = field_logits(model(t[:, :-1])[0].float(), 1, mask)
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1),
                                 reduction="none").view(len(t), -1)
            tot.index_add_(0, slot, ce.sum(0))
            cnt.index_add_(0, slot, torch.full_like(slot, len(t), dtype=torch.float32))
            if eyes is not None:
                e = torch.from_numpy(eyes[i:i + batch, 1:]).to(dev)
                eye_tot += float(ce[e].sum())
                eye_cnt += float(e.sum())
    strokes = slice(0, len(SLOTS) - 1)
    out = {"all": float(tot[strokes].sum() / cnt[strokes].sum())}
    out.update({s: float(tot[i] / cnt[i]) for i, s in enumerate(SLOTS) if cnt[i] > 0})
    if eyes is not None:
        out["eye"] = eye_tot / max(eye_cnt, 1.0)
    return out


class Watchdog:
    """Stops a GPU loop that slows down as it goes (the painter once went from 32 s to 992 s a cat as MPS memory
    piled up, devlog 2026-10-10 §10.7): the median of the last `window` timings may not exceed the median of the
    first `baseline` by more than `factor`. Every timing is logged with the GPU memory in use."""

    def __init__(self, name: str, baseline: int = 3, window: int = 3, factor: float = 2.5):
        self.name, self.baseline, self.window, self.factor = name, baseline, window, factor
        self.times: list[float] = []

    def tick(self, seconds: float, dev: str) -> dict[str, float | str]:
        import statistics

        import torch
        self.times.append(seconds)
        rec = {"watch": self.name, "seconds": round(seconds, 2)}
        if dev == "cuda":
            rec["mem_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
            rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        elif dev == "mps":
            rec["mem_gb"] = round(torch.mps.current_allocated_memory() / 2**30, 2)
        if len(self.times) >= self.baseline + self.window:
            base = statistics.median(self.times[:self.baseline])
            recent = statistics.median(self.times[-self.window:])
            rec["slowdown"] = round(recent / base, 2)
            if recent > base * self.factor:
                raise RuntimeError(f"{self.name} slowed down: recent median {recent:.1f}s is over {self.factor}× the "
                                   f"first {self.baseline} timings' {base:.1f}s; stopped before wasting GPU time")
        return rec


def autocast(dev: str) -> contextlib.AbstractContextManager:
    """fp16 autocast on CUDA (T4s have no bf16), nothing elsewhere."""
    import torch
    return torch.autocast("cuda", dtype=torch.float16) if dev == "cuda" else contextlib.nullcontext()


def _lr(cfg: Config, step: int) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = min(1.0, (step - cfg.warmup) / max(1, cfg.max_steps - cfg.warmup))
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))


class Batches:
    """Endless shuffled batches of training rows and their per-token loss weights: each pass in a new order."""

    def __init__(self, rows: np.ndarray, batch: int, seed: int, weights: np.ndarray | None = None):
        self.rows, self.batch, self.rng = rows, batch, np.random.default_rng(seed)
        self.weights = np.ones(rows.shape, np.float32) if weights is None else weights
        self.order, self.pos = self.rng.permutation(len(rows)), 0

    def next(self) -> tuple[np.ndarray, np.ndarray]:
        if self.pos + self.batch > len(self.order):
            self.order, self.pos = self.rng.permutation(len(self.rows)), 0
        self.pos += self.batch
        ix = self.order[self.pos - self.batch:self.pos]
        return self.rows[ix], self.weights[ix]


GRAD_CLIP = 1.0
BETAS = (0.9, 0.95)


def _optimiser(model, cfg: Config):
    import torch
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": other, "weight_decay": 0.0}], lr=cfg.lr, betas=BETAS)


def _step(model, opt, scaler, batches: Batches, cfg: Config, step: int, dev: str, mask=None) -> float:
    """One optimiser step over cfg.accum batches; returns the mean training loss."""
    import torch
    model.train()
    for g in opt.param_groups:
        g["lr"] = _lr(cfg, step)
    total = 0.0
    for _ in range(cfg.accum):
        rows, weights = batches.next()
        t, w = torch.from_numpy(rows).to(dev), torch.from_numpy(weights[:, 1:]).to(dev)
        with autocast(dev):
            logits = field_logits(model(t[:, :-1])[0], 1, mask)
            ce = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1),
                                                   reduction="none").view_as(w)
            loss = (ce * w).sum() / w.sum()
        scaler.scale(loss / cfg.accum).backward()
        total += loss.item() / cfg.accum
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    return total


def train(cfg: Config, data: Path = DATA_FILE, run_dir: Path = RUN_DIR) -> Path:
    """Train on the train split (and its mirror images); every eval_every optimiser steps, measure the loss on the
    validation split (never trained on, not mirrored) and keep the weights with the lowest one. Stops early after
    cfg.patience evaluations without a better one, after cfg.stop_after_min minutes, or when a stretch of steps
    gets much slower than the first ones (Watchdog)."""
    import torch
    from ..llm.model import GPT
    dev = device()
    out = run_dir / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    train_rows, train_ids = load_rows(data, "train", cfg.cats, cfg.order, points=cfg.points)
    train_pts = load_points(data, "train", cfg.cats)
    val_rows, _ = load_rows(data, "val", 0, cfg.order, points=cfg.points)
    val_eyes = eye_strokes(val_rows, load_points(data, "val"))
    if cfg.mirror:
        train_rows = np.concatenate([train_rows, load_rows(data, "train", cfg.cats, cfg.order, mirrored=True,
                                                           points=cfg.points)[0]])
        train_pts = np.concatenate([train_pts, load_points(data, "train", cfg.cats, mirrored=True)])
    weights = np.where(eye_strokes(train_rows, train_pts), cfg.eye_weight, 1.0).astype(np.float32)
    mask = allowed(dev, cfg.points) if cfg.heads else None
    torch.manual_seed(cfg.seed)
    model = GPT(model_config(cfg)).to(dev)
    opt, scaler = _optimiser(model, cfg), torch.amp.GradScaler(enabled=dev == "cuda")
    info = {"config": asdict(cfg), "model": asdict(model_config(cfg)), "parameters": model.num_params(),
            "train_cats": len(train_ids), "train_rows": len(train_rows), "val_rows": len(val_rows),
            "tokens_per_step": cfg.batch * cfg.accum * (train_rows.shape[1] - 1), "device": dev,
            "eye_token_share": round(float((weights > 1).mean()) if cfg.eye_weight != 1 else 0.0, 4),
            "gpu": torch.cuda.get_device_name() if dev == "cuda" else ""}
    (out / "config.json").write_text(json.dumps(info, indent=1) + "\n")
    (out / "train_ids.txt").write_text("\n".join(train_ids) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "config"}), flush=True)
    log = out / "metrics.jsonl"
    log.write_text("")
    batches, watch = Batches(train_rows, cfg.batch, cfg.seed, weights), Watchdog("train interval")
    best, since_best, losses, t0, t_last = float("inf"), 0, [], time.time(), time.time()
    for step in range(cfg.max_steps):
        losses.append(_step(model, opt, scaler, batches, cfg, step, dev, mask))
        late = cfg.stop_after_min and time.time() - t0 > cfg.stop_after_min * 60
        if (step + 1) % cfg.eval_every and step + 1 < cfg.max_steps and not late:
            continue
        train_s = time.time() - t_last
        val = slot_losses(model, val_rows, dev, mask=mask, eyes=val_eyes)
        rec = {"step": step + 1, "epoch": round((step + 1) * cfg.batch * cfg.accum / len(train_rows), 2),
               "train_loss": round(float(np.mean(losses)), 4), "val_loss": round(val["all"], 4),
               "val_slots": {k: round(v, 4) for k, v in val.items() if k != "all"},
               "lr": round(_lr(cfg, step), 7), "tokens_per_s": round(len(losses) * info["tokens_per_step"] / train_s),
               "minutes": round((time.time() - t0) / 60, 1),
               **watch.tick(train_s / len(losses), dev)}
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        losses = []
        if val["all"] < best:
            best, since_best = val["all"], 0
            torch.save({"model": model.state_dict(), "config": asdict(cfg), **rec}, out / "best.pt")
        else:
            since_best += 1
        if since_best >= cfg.patience or late:
            print(json.dumps({"stopped": "patience" if since_best >= cfg.patience else "time", "step": step + 1}),
                  flush=True)
            break
        t_last = time.time()
    return out / "best.pt"


def load_model(path: Path, dev: str = "cpu") -> "GPT":
    """The GPT saved in a run's best.pt, in eval mode on dev."""
    import torch
    from ..llm.model import GPT
    st = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(model_config(Config(**st["config"])))
    model.load_state_dict(st["model"])
    model.brush_cfg = Config(**st["config"])                  # sample() reads `heads` from it
    return model.to(dev).eval()


def sample(model: "GPT", n: int, temperature: float = 1.0, seed: int = 0, batch: int = 64) -> np.ndarray:
    """n new paintings, (n, SEQ) tokens, sampled from BOS one token at a time with a preallocated KV cache. With one
    output layer per field (the model's config `heads`), each position samples among its field's tokens only;
    otherwise among all tokens, with nothing removed."""
    import torch
    dev = next(model.parameters()).device.type
    cfg = getattr(model, "brush_cfg", None) or Config()
    mask = allowed(dev, cfg.points) if cfg.heads else None
    seq = SEQ + (PREFIX if cfg.points else 0)
    torch.manual_seed(seed)
    out, watch = [], Watchdog("sample batch", baseline=1, window=1)
    for i in range(0, n, batch):
        t0 = time.time()
        b = min(batch, n - i)
        rows = torch.full((b, seq), T.BOS, dtype=torch.long, device=dev)
        cache = [{"size": seq} for _ in model.blocks]
        with torch.no_grad(), autocast(dev):
            logits, _ = model(rows[:, :1], cache=cache)
            for p in range(1, seq):
                last = field_logits(logits[:, -1:].float(), p, mask)[:, 0] / temperature
                rows[:, p] = torch.multinomial(last.softmax(-1), 1)[:, 0]
                if p < seq - 1:
                    logits, _ = model(rows[:, p:p + 1], cache=cache, start_pos=p)
        out.append(rows.cpu().numpy())
        print(json.dumps(watch.tick(time.time() - t0, dev)), flush=True)   # a batch takes the same time at any size
    return np.concatenate(out)


def render_all(rows: np.ndarray, dev: str, chunk: int = 50) -> np.ndarray:
    """Renders of token rows, (n, SIDE, SIDE, 3), timed in chunks by a watchdog."""
    import torch
    if not len(rows):
        return np.zeros((0, B.SIDE, B.SIDE, 3))
    out, watch = [], Watchdog("render chunk")
    for i in range(0, len(rows), chunk):
        t0 = time.time()
        out += [render(r, dev) for r in rows[i:i + chunk]]
        if dev == "cuda":
            torch.cuda.empty_cache()
        watch.tick((time.time() - t0) / len(rows[i:i + chunk]), dev)
    return np.stack(out)


def small(imgs: np.ndarray, side: int = 64) -> np.ndarray:
    """(n, SIDE, SIDE, 3) → (n, side, side, 3) by averaging blocks."""
    f = imgs.shape[1] // side
    return imgs.reshape(len(imgs), side, f, side, f, 3).mean((2, 4))


def nearest(a: np.ndarray, b: np.ndarray, skip_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """For each image of a, the mean absolute pixel difference to its nearest image of b, and which one."""
    fa, fb = a.reshape(len(a), -1), b.reshape(len(b), -1)
    dist = np.stack([np.abs(fb - x).mean(1) for x in fa])
    if skip_self:
        np.fill_diagonal(dist, np.inf)
    return dist.min(1), dist.argmin(1)


def fid(a: np.ndarray, b: np.ndarray, dev: str) -> float | None:
    """Fréchet Inception distance between two sets of (n, H, W, 3) images in 0–1, with pytorch-fid's Inception
    weights; None if pytorch-fid is not installed."""
    try:
        import torch
        from pytorch_fid.inception import InceptionV3
        from scipy import linalg
    except ImportError:
        return None
    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()

    def feats(x: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 50):
                t = torch.from_numpy(x[i:i + 50]).permute(0, 3, 1, 2).float().to(dev)
                out.append(net(t)[0].squeeze(-1).squeeze(-1).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    fa, fb = feats(a), feats(b)
    # ‖μa − μb‖² + Tr(Σa + Σb − 2(ΣaΣb)^½); computed here because pytorch-fid's helper passes an argument
    # (disp) that newer scipy no longer takes (crashed run C on 2026-10-11)
    mu, sa, sb = fa.mean(0) - fb.mean(0), np.cov(fa, rowvar=False), np.cov(fb, rowvar=False)
    root = linalg.sqrtm(sa @ sb)
    return float(mu @ mu + np.trace(sa + sb - 2 * np.real(root)))


def save_sheet(imgs: np.ndarray, path: Path, cols: int = 8) -> None:
    from PIL import Image
    rows = (len(imgs) + cols - 1) // cols
    side = imgs.shape[1]
    sheet = np.ones((rows * side, cols * side, 3))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet[r * side:(r + 1) * side, c * side:(c + 1) * side] = im
    Image.fromarray((sheet.clip(0, 1) * 255).astype(np.uint8)).save(path)


def evaluate(name: str, n: int = 256, temperature: float = 1.0, data: Path = DATA_FILE,
             run_dir: Path = RUN_DIR) -> dict[str, object]:
    """Sample n paintings from the best weights and measure:
    - how many decode: every token inside the range its position allows (no masking during sampling);
    - how close each is to its nearest training painting or its mirror image (64-pixel renders, mean absolute
      difference), next to the
      same distance for the validation paintings, which are real paintings the model never saw: generated ones far
      closer than those would mean copying;
    - FID against the training paintings' renders, next to the validation paintings' FID against the same set
      (different sample sizes: comparable between runs of the same n, not an absolute quality number)."""
    import torch
    dev = device()
    out = run_dir / name
    model = load_model(out / "best.pt", dev)
    t0 = time.time()
    gen = sample(model, n, temperature)
    sample_s = time.time() - t0
    ok = valid(gen)
    good = gen[ok.all(1)]
    t0 = time.time()
    train_rows, _ = load_rows(data, "train", Config(**torch.load(out / "best.pt", map_location="cpu",
                                                                  weights_only=False)["config"]).cats)
    val_rows, _ = load_rows(data, "val")
    gen_img, train_img, val_img = (render_all(r, dev) for r in (good, train_rows, val_rows))
    render_s = time.time() - t0
    g64, v64 = small(gen_img), small(val_img)
    # the model trains on mirror images too, so a copy may be a mirrored one: compare with both
    both = np.concatenate([train_img, train_img[:, :, ::-1]])
    t64 = small(both)
    report = {"name": name, "n": n, "temperature": temperature, "sample_seconds": round(sample_s, 1),
              "render_seconds": round(render_s, 1),
              "decoded": int(ok.all(1).sum()), "decoded_rate": round(float(ok.all(1).mean()), 4),
              "token_valid_rate": round(float(ok.mean()), 6),
              "eos_at_end": round(float((gen[:, -1] == T.EOS).mean()), 4)}
    vd, _ = nearest(v64, t64)
    report["val_nearest_train"] = {"median": round(float(np.median(vd)), 4), "min": round(float(vd.min()), 4)}
    if len(good):
        gd, gi = nearest(g64, t64)
        report["gen_nearest_train"] = {"median": round(float(np.median(gd)), 4), "min": round(float(gd.min()), 4),
                                       "below_val_min": int((gd < vd.min()).sum())}
        save_sheet(gen_img[:32], out / "samples.png")
        # the painting process: each of 8 samples after stage 1, 2, 3 and 4 (one row per sample)
        steps = [T.render(T.decode(strokes_only(r).tolist())[:k], dev) for r in good[:8] for k in range(1, STAGES + 1)]
        save_sheet(np.stack(steps), out / "process.png", cols=STAGES)
        close = np.argsort(gd)[:8]
        pairs = np.concatenate([np.stack([gen_img[i], both[gi[i]]]) for i in close])
        save_sheet(pairs, out / "closest_pairs.png", cols=4)
        if len(good) > 1:
            dd, _ = nearest(g64, g64, skip_self=True)
            report["gen_nearest_gen_median"] = round(float(np.median(dd)), 4)
        report["fid_gen_vs_train"] = fid(gen_img, train_img, dev)
    report["fid_val_vs_train"] = fid(val_img, train_img, dev)
    report["sizes"] = {"gen": len(good), "train": len(train_img), "val": len(val_img)}
    (out / "eval.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main() -> None:
    """Command line: see the module docstring."""
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "align", "train", "eval"])
    ap.add_argument("--name", default="trial")
    ap.add_argument("--data", type=Path, default=DATA_FILE)
    ap.add_argument("--runs", type=Path, default=RUN_DIR)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--paint-dirs", type=Path, nargs="+", default=[PAINT_DIR], help="data: the painting folders")
    ap.add_argument("--aligned", type=Path, default=ALIGNED_FILE, help="align: the output file (input: --data)")
    ap.add_argument("--align-ref", type=Path, default=None, help="align: take the eye midpoint and distance from here")
    for f, v in asdict(Config()).items():
        if f != "name":
            ap.add_argument("--" + f.replace("_", "-"), type=type(v) if not isinstance(v, bool) else int, default=v)
    a = ap.parse_args()
    if a.stage == "data":
        stage_data(a.paint_dirs if len(a.paint_dirs) > 1 else a.paint_dirs[0], out=a.data)
    elif a.stage == "align":
        align_data(a.data, a.aligned, a.align_ref)
    elif a.stage == "train":
        cfg = Config(name=a.name, **{f: getattr(a, f) for f in asdict(Config()) if f != "name"})
        cfg.mirror, cfg.heads, cfg.points = bool(cfg.mirror), bool(cfg.heads), bool(cfg.points)
        train(cfg, a.data, a.runs)
    else:
        evaluate(a.name, a.n, a.temperature, a.data, a.runs)


if __name__ == "__main__":
    main()
