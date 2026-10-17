"""What each AFHQ cat looks like, in words a condition token can carry: fur colours, which part of the face has a
patch of another colour, each eye's colour, and the coat pattern. CPU only; no model is loaded.

Every position is relative to the cat's own face, never to pixels, so the same description fits a photo, a painting
and a player's doodle of any size. The face frame comes from the nine face points (oilprocess.Face):

    origin  the midpoint of the two eyes          unit  d, the distance between the eyes
    u       along Face.right (eye 0 → eye 1)      v     along Face.up (mouth → eye midpoint)

Left and right are the PICTURE's left and right (左右以畫面為準), as in cat-face-landmarks' NAMES (left_eye,
right_eye, mouth, left_ear_outer, left_ear_tip, left_ear_inner, right_ear_inner, right_ear_tip, right_ear_outer):
point 0 (left_eye) is left of point 1 in all 5,558 cats, so "left" = −u = the left half of the picture (on a cat
facing the camera, that is its own right eye). Brush GPT's mirror augmentation swaps MIRROR_POINTS
[1, 0, 2, 8, 7, 6, 5, 4, 3]; `mirror` swaps these descriptions the same way (left ↔ right keys, crown grid flipped).
Typical face points (medians over all 5,558 cats, in (u, v)): eyes (∓0.5, 0), mouth (0, −0.81), estimated nose
(0, −0.57), inner ear bases (∓0.47, 0.96), ear tips (∓1.25, 1.6), outer ear bases (∓1.15, 0.48).

The photo is the canvas Brush GPT learns from: catbrush.load with the painting's own box (256 × 256, white paper
outside the subject mask); cats that were never painted use paint_afhq.face_box, as the painter would.
Colours are measured in CIE Lab (L 0–100, a green→red, b blue→yellow; chroma C = √(a² + b²), hue h = atan2(b, a)).

1  Fur palette. Fur = mask > 0.9, minus a 0.34 d disc around each eye and a 0.14 d disc around the estimated nose
   (the pink nose leather is not fur). The Lab image is smoothed (σ 1.2 px) and k-means is run for k = 3, 4, 5 on
   (0.6 L, a, b) (L down-weighted so shading splits clusters less than real colour does); the k with the best
   silhouette is kept and stored with each cluster's share and name.
   Base colour (base_colour): the clusters are split into dark (L < 25), white (L ≥ 68, C < 12) and the rest.
       black   dark ≥ 45 % of the fur                     white   white ≥ 45 %
       otherwise, from the share-weighted mean of the rest (Lr, Cr, hr):
       grey    Cr < 12.5, or a hue outside 20°–110° (cool, bluish grey)
       orange  Cr ≥ 24 and every rest cluster darker than L 60 still has C ≥ 20 (orange tabby: even the stripes
               are orange; brown tabby: the stripes are nearly black)
       cream   Lr ≥ 62          brown   everything else
   The main fur colour (main_lab) is the mean of the clusters that made the base colour. A single colour (a region,
   a patch, a cluster) is named with the same thresholds (fur_name). The main colour category is "mixed" when the
   pattern (5) is bicolour or tortie/calico, else the base colour. brown is not in the requested list (white,
   black, orange, grey, cream, mixed); it is kept because brown tabbies are a large part of AFHQ.
   Colours are the photo's own pixels, not corrected for light: a white cat photographed in dim warm light is
   measured, and painted, as brownish grey, and is named that way. The painting has the same colours as the photo,
   so the label describes what the painting shows.

2  Face regions, each a shape in (u, v) cut by the fur mask (eye and nose discs removed):
       crown         between the ears, above the eyes: polygon (−0.5, 0.3) → inner ear base 5 → 0.35 d above it →
                     0.35 d above inner ear base 6 → inner ear base 6 → (0.5, 0.3), minus the ear triangles
       ear_left      triangle of points 3, 4, 5         ear_right     triangle of points 6, 7, 8
       bridge        |u| ≤ 0.17, −0.32 ≤ v ≤ 0.3, outside the muzzle
       muzzle_chin   ellipse centred (0, −0.72), half-axes 0.42 (u) and 0.40 (v): nose, mouth and chin
       cheek_left    −1.15 ≤ u ≤ −0.25, −0.95 ≤ v ≤ −0.2, outside the muzzle      cheek_right   its mirror
       eye_ring_left the fur around eye 0 from 0.34 d (where the eye disc that fur leaves out ends) to 0.6 d,
                     outside the ears: white "spectacles", dark rims, a patch over one eye     eye_ring_right  eye 1
                     The rings overlap the crown, bridge and cheeks; every region is measured on its own.
   For each region: mean Lab, ΔE (CIE76) from the main fur colour, and its patch: pixels at least 0.05 d inside the
   subject mask (its edge mixes in white paper and backlit fur) whose colour (Lab smoothed with σ = 0.1 d, so
   stripes blur away) differs from the main fur colour by
       D = √((0.5 ΔL)² + Δa² + Δb²) ≥ PATCH_D
   (a difference of lightness alone must be twice as large as one of hue: shading is mostly lightness), opened by a
   3 px disc, connected parts smaller than 0.02 d² dropped. "has_patch" when the patch covers ≥ PATCH_SHARE of the
   region; the patch colour is the patch pixels' mean Lab and its fur colour name.

3  Crown patch shape: the crown's patch mask sampled on a 16 × 16 grid over u ∈ [−0.7, 0.7], v ∈ [0.25, 1.45].
   Row 0 is the top (v = 1.45), column 0 is the picture's left, so the grid looks like the photo.
   Each cell is the share of crown pixels in it that are patch (0–1), or −1 where the cell holds no crown.

4  Eyes: pixels within 0.2 d of each eye point, the darkest 35 % (pupil) and brightest 5 % (glints) dropped, mean
   Lab of the rest. Names (EYE_NAMES): unclear when L < 22 (closed, shadowed or the point missed) or C < 8;
   blue h ≥ 180° or h < −30° (b < 0); green 105° ≤ h < 180°; yellow 75° ≤ h < 105°; then orange when L ≥ 55,
   copper when darker. Odd eyes (heterochromia): both eyes named (not unclear), different names, ΔE ≥ 25, and one
   of them blue.

5  Pattern (PATTERNS), from two kinds of measurement:
       stripes  dog = L blurred at σ 0.025 d minus L blurred at σ 0.1 d (a band at the scale of stripes; the head's
                shading is coarser and drops out). stripe = std(dog) over the head's fur (eyes, ears and muzzle
                left out) + std(dog) over the crown. Dark marks = dog < −6, parts ≥ 0.004 d²; mark_share is their
                area share, elongated_share the share of their area in parts with major/minor axis ≥ 2.5.
       blocks   the face (ears, eyes and nose left out) cut into 0.4 d × 0.4 d cells and each cell's median Lab:
                larger than a stripe, so stripes average out, smaller than a patch, so patches stay.
                cell_white L ≥ 62 and C < 12; cell_black L < 28; cell_warm C ≥ 15, 30° ≤ h ≤ 95°, L ≥ 30 (each a
                share of the cells); cell_l_range = 90th − 10th percentile of the cells' L; cell_c90 the 90th
                percentile of their chroma.
   tortie/calico  calico: cell_white ≥ 0.15, cell_black ≥ 0.08 and cell_warm ≥ 0.08; or tortie: cell_black ≥ 0.1,
                  0.1 ≤ cell_warm ≤ 0.35, cell_c90 < 20 and cell_white < 0.15
   bicolour       cell_white ≥ 0.15 and cell_l_range ≥ 40
   tabby          stripe ≥ 10.5 and mark_share ≥ 0.05
   spots/blotches as tabby, but elongated_share < 0.15
   solid          everything else
   The thresholds were set by eye on 48 hand-labelled cats (runs/coat/handcheck, set A) and measured once on 48
   others (set B) that were labelled before any result was seen; --report prints both.

    python tools/coat_attributes.py                 # every cat, writes runs/coat/
    python tools/coat_attributes.py --limit 200     # a quick look
    python tools/coat_attributes.py --report        # classify runs/coat/measures.jsonl again, table and mosaics
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
sys.path.insert(0, str(ML / "tools"))

LABELS = ML / "data/processed/afhq_cats_mae_pad1.5.npz"
AFHQ, MASKS, PAINTED = ML / "data/raw/afhq", ML / "data/processed/afhq_masks", ML / "data/processed/afhq_brush_face"
OUT = ML / "runs/coat"
SIDE = 256

EYE_R, NOSE_R, IRIS_R, RING_R = 0.34, 0.14, 0.2, 0.6
PATCH_D, PATCH_SHARE = 18.0, 0.2
PATCH_BLUR, PATCH_EDGE = 0.1, 0.05     # in eye distances
DARK_L, WHITE_L, WHITE_C = 25.0, 68.0, 12.0
GREY_C, ORANGE_C, ORANGE_DARK_C, CREAM_L = 12.5, 24.0, 20.0, 62.0
CELL = 0.4                       # side of the face cells, in eye distances: larger than a stripe, smaller than a patch
STRIPE, MARKS, SPOT_ELONG = 10.5, 0.05, 0.15
ODD_DE = 25.0
GRID_N, GRID_U, GRID_V = 16, (-0.7, 0.7), (0.25, 1.45)

FUR_NAMES = ["white", "black", "grey", "orange", "cream", "brown"]
EYE_NAMES = ["yellow", "green", "blue", "orange", "copper", "unclear"]
PATTERNS = ["solid", "tabby", "spots", "bicolour", "tortie_calico"]
REGIONS = ["crown", "ear_left", "ear_right", "bridge", "muzzle_chin", "cheek_left", "cheek_right", "eye_ring_left",
           "eye_ring_right"]
ZH = {"white": "白", "black": "黑", "grey": "灰", "orange": "橘", "cream": "奶油", "brown": "棕", "mixed": "混色",
      "yellow": "黃", "green": "綠", "blue": "藍", "copper": "銅", "unclear": "看不清",
      "solid": "純色", "tabby": "虎斑條紋", "spots": "斑點/塊狀", "bicolour": "雙色", "tortie_calico": "玳瑁/三花",
      "crown": "頭頂", "ear_left": "左耳", "ear_right": "右耳", "bridge": "鼻樑", "muzzle_chin": "口鼻下巴",
      "cheek_left": "左臉頰", "cheek_right": "右臉頰",
      "eye_ring_left": "左眼周", "eye_ring_right": "右眼周"}


# ---------------------------------------------------------------- colour names

def lch(lab: np.ndarray) -> tuple[float, float, float]:
    L, a, b = (float(x) for x in lab)
    return L, math.hypot(a, b), math.degrees(math.atan2(b, a))


def fur_name(lab: np.ndarray) -> str:
    """One colour's name, with the same thresholds as `base_colour`."""
    L, C, h = lch(lab)
    if L < DARK_L:
        return "black"
    if L >= WHITE_L and C < WHITE_C:
        return "white"
    if C < GREY_C or not 20 <= h <= 110:
        return "grey"
    if C >= ORANGE_C:
        return "orange"
    return "cream" if L >= CREAM_L else "brown"


def eye_name(lab: np.ndarray) -> str:
    L, C, h = lch(lab)
    if L < 22 or C < 8:
        return "unclear"
    if h >= 180 or h < -30:
        return "blue"
    if 105 <= h < 180:
        return "green"
    if 75 <= h < 105:
        return "yellow"
    return "orange" if L >= 55 else "copper"


def delta_e(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(p, float) - np.asarray(q, float)))


# ---------------------------------------------------------------- geometry

def face_uv(face) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) of every canvas pixel centre in the face frame."""
    ys, xs = np.mgrid[0:SIDE, 0:SIDE] + 0.5
    mid = ((face.eye_l + face.eye_r) / 2).numpy()
    right, up = face.right.numpy(), face.up.numpy()
    rx, ry = xs - mid[0], ys - mid[1]
    return (rx * right[0] + ry * right[1]) / face.d, (rx * up[0] + ry * up[1]) / face.d


def to_uv(face, pts: np.ndarray) -> np.ndarray:
    mid = ((face.eye_l + face.eye_r) / 2).numpy()
    rel = np.asarray(pts, float) - mid
    return np.stack([rel @ face.right.numpy(), rel @ face.up.numpy()], -1) / face.d


def in_polygon(u: np.ndarray, v: np.ndarray, poly: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path as MPath
    pts = np.stack([u.ravel(), v.ravel()], 1)
    return MPath(poly).contains_points(pts).reshape(u.shape)


def regions(face, u: np.ndarray, v: np.ndarray, fur: np.ndarray) -> dict[str, np.ndarray]:
    P = to_uv(face, face.p.numpy())
    ear_l = in_polygon(u, v, P[[3, 4, 5]])            # picture left
    ear_r = in_polygon(u, v, P[[6, 7, 8]])
    lift = np.array([0, 0.35])
    crown = in_polygon(u, v, np.array([[-0.5, 0.3], P[5], P[5] + lift, P[6] + lift, P[6], [0.5, 0.3]]))
    crown &= ~ear_r & ~ear_l
    muzzle = (u / 0.42) ** 2 + ((v + 0.72) / 0.40) ** 2 <= 1
    bridge = (np.abs(u) <= 0.17) & (v >= -0.32) & (v <= 0.3) & ~muzzle
    cheek = (np.abs(u) >= 0.25) & (np.abs(u) <= 1.15) & (v >= -0.95) & (v <= -0.2) & ~muzzle
    out = {"crown": crown, "ear_left": ear_l, "ear_right": ear_r, "bridge": bridge, "muzzle_chin": muzzle,
           "cheek_left": cheek & (u < 0), "cheek_right": cheek & (u > 0)}
    for key, (eu, ev) in (("eye_ring_left", P[0]), ("eye_ring_right", P[1])):
        r = np.hypot(u - eu, v - ev)
        out[key] = (r > EYE_R) & (r <= RING_R) & ~ear_l & ~ear_r
    return {k: r & fur for k, r in out.items()}


# ---------------------------------------------------------------- one cat

def load_cat(i: int, labels) -> tuple:
    from paint_afhq import face_box
    from sellanything_ml.sketch import catbrush
    from sellanything_ml.sketch.oilprocess import Face
    rel = str(labels["ids"][i])
    stem = Path(rel).stem
    painted = (PAINTED / f"{stem}.npz").exists()
    box = tuple(int(x) for x in np.load(PAINTED / f"{stem}.npz")["box"]) if painted else face_box(labels["points"][i])
    photo = catbrush.load(str(AFHQ / rel), str(MASKS / f"{stem}_mask1.png"), box)
    face = Face.from_photo(labels["points"][i], box)
    return rel, stem, painted, box, photo.target.numpy(), photo.mask.numpy(), face


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 3, iters: int = 40) -> np.ndarray:
    """Centres of k-means (k-means++ start, Lloyd steps), best of n_init by inertia. Plain numpy: sklearn's OpenMP
    and torch's crash each other in one process on this Mac."""
    best = None
    for _ in range(n_init):
        c = [x[rng.integers(len(x))]]
        for _ in range(k - 1):
            d2 = ((x[:, None] - np.array(c)[None]) ** 2).sum(-1).min(1)
            c.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        c = np.array(c)
        for _ in range(iters):
            lab = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
            new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = ((x - c[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, c)
    return best[1]


def silhouette(x: np.ndarray, lab: np.ndarray) -> float:
    dist = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
    ks = np.unique(lab)
    if len(ks) < 2:
        return -1.0
    mean_to = np.stack([dist[:, lab == j].mean(1) for j in ks], 1)          # (n, k)
    own = np.searchsorted(ks, lab)
    cnt = np.array([(lab == j).sum() for j in ks])[own]
    a = mean_to[np.arange(len(x)), own] * cnt / np.maximum(cnt - 1, 1)
    mean_to[np.arange(len(x)), own] = np.inf
    b = mean_to.min(1)
    return float(np.nan_to_num((b - a) / np.maximum(a, b)).mean())


def palette(lab_px: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster centres (real Lab), shares, and every pixel's cluster, best k by silhouette."""
    w = np.array([0.6, 1, 1])
    take = rng.choice(len(lab_px), min(4000, len(lab_px)), replace=False)
    x = lab_px[take] * w
    sub = x[rng.choice(len(x), min(1500, len(x)), replace=False)]
    best = None
    for k in (3, 4, 5):
        c = kmeans(x, k, rng)
        s = silhouette(sub, ((sub[:, None] - c[None]) ** 2).sum(-1).argmin(1))
        if best is None or s > best[0]:
            best = (s, c)
    c = best[1]
    xa = lab_px * w
    lab_all = np.concatenate([((xa[s:s + 20000, None] - c[None]) ** 2).sum(-1).argmin(1)
                              for s in range(0, len(xa), 20000)])
    k = len(c)
    centres = np.array([lab_px[lab_all == j].mean(0) if (lab_all == j).any() else c[j] / w for j in range(k)])
    shares = np.bincount(lab_all, minlength=k) / len(lab_all)
    order = np.argsort(-shares)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return centres[order], shares[order], remap[lab_all]


def base_colour(centres: np.ndarray, shares: np.ndarray) -> tuple[str, np.ndarray, dict]:
    """The base fur colour from the palette clusters, its Lab (the share-weighted mean of the clusters that make it),
    and the shares used. Clusters are first split into dark (L < 25), white (L ≥ 68, C < 12) and the rest."""
    L = centres[:, 0]
    C = np.hypot(centres[:, 1], centres[:, 2])
    dark, white = L < DARK_L, (L >= WHITE_L) & (C < WHITE_C)
    rest = ~dark & ~white
    s = {"dark": float(shares[dark].sum()), "white": float(shares[white].sum()), "rest": float(shares[rest].sum())}

    def mean(sel: np.ndarray) -> np.ndarray:
        return (centres[sel] * shares[sel, None]).sum(0) / shares[sel].sum()

    if s["dark"] >= 0.45 or (s["rest"] < 0.2 and s["dark"] >= s["white"]):
        return "black", mean(dark), s
    if s["white"] >= 0.45 or s["rest"] < 0.2:
        return "white", mean(white), s
    m = mean(rest)
    Lr, Cr, hr = lch(m)
    if Cr < GREY_C or not 20 <= hr <= 110:
        return "grey", m, s
    darker = rest & (L < 60)
    if Cr >= ORANGE_C and (not darker.any() or C[darker].min() >= ORANGE_DARK_C):
        return "orange", m, s
    return ("cream" if Lr >= CREAM_L else "brown"), m, s


def classify(row: dict) -> dict:
    """Every label from the stored measurements (thresholds at the top of the file)."""
    centres = np.array([p["lab"] for p in row["palette"]])
    for p in row["palette"]:
        p["name"] = fur_name(np.array(p["lab"]))
    base, _, _ = base_colour(centres, np.array([p["share"] for p in row["palette"]]))
    f = row["pattern_features"]
    stripe = f["dog_head"] + f["dog_crown"]
    W, K, warm = f["cell_white"], f["cell_black"], f["cell_warm"]
    if W >= 0.15 and K >= 0.08 and warm >= 0.08:
        pattern = "tortie_calico"                     # calico: white, black and warm blocks
    elif K >= 0.1 and 0.1 <= warm <= 0.35 and f["cell_c90"] < 20 and W < 0.15:
        pattern = "tortie_calico"                     # tortie: black and muted warm blocks, no white
    elif W >= 0.15 and f["cell_l_range"] >= 40:
        pattern = "bicolour"
    elif stripe >= STRIPE and f["mark_share"] >= MARKS and f["elongated_share"] < SPOT_ELONG:
        pattern = "spots"
    elif stripe >= STRIPE and f["mark_share"] >= MARKS:
        pattern = "tabby"
    else:
        pattern = "solid"
    row["base_colour"] = base
    row["main_category"] = "mixed" if pattern in ("bicolour", "tortie_calico") else base
    row["pattern"] = pattern
    for r in row["regions"].values():
        if r["visible"]:
            r["name"] = fur_name(np.array(r["mean_lab"]))
            r["has_patch"] = r["patch_share"] >= PATCH_SHARE
            if "patch_lab" in r:
                r["patch_name"] = fur_name(np.array(r["patch_lab"]))
    for key in ("eye_left", "eye_right"):
        lab = row["eyes_lab"][key]
        row[key] = {"name": eye_name(np.array(lab)) if lab else "unclear", "lab": lab}
    er, el = row["eye_left"], row["eye_right"]
    both = er["lab"] is not None and el["lab"] is not None
    row["eye_delta_e"] = round(delta_e(er["lab"], el["lab"]), 1) if both else None
    row["odd_eyes"] = bool(both and "unclear" not in (er["name"], el["name"]) and er["name"] != el["name"]
                           and "blue" in (er["name"], el["name"]) and row["eye_delta_e"] >= ODD_DE)
    return row


def describe(i: int) -> dict:
    """The measurements of cat i (labels come later, from `classify`)."""
    import torch
    from scipy import ndimage as ndi
    from scipy.ndimage import map_coordinates
    from skimage.color import rgb2lab
    from skimage.measure import label as cc_label
    from skimage.measure import regionprops
    from skimage.morphology import disk
    torch.set_num_threads(1)
    labels = np.load(LABELS)
    rel, stem, painted, box, img, m, face = load_cat(i, labels)
    rng = np.random.default_rng(i)
    d = face.d
    u, v = face_uv(face)
    lab = rgb2lab(img.clip(0, 1))
    lab_s = np.stack([ndi.gaussian_filter(lab[..., c], 1.2) for c in range(3)], -1)
    eyes = [face.eye_l.numpy(), face.eye_r.numpy()]                 # point 0 (picture left), point 1 (picture right)
    ys, xs = np.mgrid[0:SIDE, 0:SIDE] + 0.5
    dist = lambda c: np.hypot(xs - c[0], ys - c[1])                 # noqa: E731
    no_eyes = (dist(eyes[0]) > EYE_R * d) & (dist(eyes[1]) > EYE_R * d) & (dist(face.nose.numpy()) > NOSE_R * d)
    fur = (m > 0.9) & no_eyes
    if fur.sum() < 500:
        return {"id": rel, "error": f"only {int(fur.sum())} fur pixels"}
    wm = fur.astype(float)

    def nblur(x: np.ndarray, s: float) -> np.ndarray:                # blur that only averages fur pixels
        return ndi.gaussian_filter(x * wm, s) / (ndi.gaussian_filter(wm, s) + 1e-6)

    # 1 palette and base colour
    centres, shares, _ = palette(lab_s[fur], rng)
    _, main, _ = base_colour(centres, shares)

    # 5 pattern measurements
    shape0 = regions(face, u, v, np.ones_like(fur))
    no_ears = ~shape0["ear_left"] & ~shape0["ear_right"]
    head = (ndi.binary_erosion(m > 0.9, disk(4)) & no_ears & ~shape0["muzzle_chin"] & (v >= -0.95) & (v <= 1.35)
            & (np.abs(u) <= 1.15) & (dist(eyes[0]) > 0.45 * d) & (dist(eyes[1]) > 0.45 * d))
    if head.sum() < 300:
        head = fur
    crown_in = shape0["crown"] & ndi.binary_erosion(m > 0.9, disk(3))
    dog = nblur(lab[..., 0], 0.025 * d) - nblur(lab[..., 0], 0.1 * d)
    marks = ndi.binary_opening((dog < -6) & head, disk(1))
    mark_area = elong_area = 0.0
    for r in regionprops(cc_label(marks)):
        if r.area < 0.004 * d * d:
            continue
        mark_area += r.area
        if r.minor_axis_length > 0 and r.major_axis_length / r.minor_axis_length >= 2.5:
            elong_area += r.area
    cells = []
    ok = (m > 0.9) & no_ears & no_eyes
    for a in np.arange(-1.2, 1.2, CELL):
        for b in np.arange(-1.1, 1.3, CELL):
            c = ok & (u >= a) & (u < a + CELL) & (v >= b) & (v < b + CELL)
            if c.sum() > 0.6 * (CELL * d) ** 2:
                cells.append(np.median(lab[c], 0))
    cells = np.array(cells) if cells else np.zeros((1, 3))
    cl, cc_ = cells[:, 0], np.hypot(cells[:, 1], cells[:, 2])
    ch = np.degrees(np.arctan2(cells[:, 2], cells[:, 1]))
    feat = {"dog_head": float(dog[head].std()), "dog_crown": float(dog[crown_in].std()) if crown_in.sum() > 50 else 0.0,
            "mark_share": mark_area / head.sum(), "elongated_share": elong_area / mark_area if mark_area else 0.0,
            "cells": len(cells), "cell_white": float(((cl >= 62) & (cc_ < 12)).mean()),
            "cell_black": float((cl < 28).mean()),
            "cell_warm": float(((cc_ >= 15) & (ch >= 30) & (ch <= 95) & (cl >= 30)).mean()),
            "cell_l_range": float(np.percentile(cl, 90) - np.percentile(cl, 10)),
            "cell_c90": float(np.percentile(cc_, 90))}

    # 2 regions and patches
    sig = PATCH_BLUR * d
    lab_p = np.stack([nblur(lab[..., c], sig) for c in range(3)], -1)
    diff = np.sqrt((0.5 * (lab_p[..., 0] - main[0])) ** 2 + (lab_p[..., 1] - main[1]) ** 2
                   + (lab_p[..., 2] - main[2]) ** 2)
    inner = fur & ndi.binary_erosion(m > 0.9, disk(max(1, round(PATCH_EDGE * d))))   # off the mask's edge
    patch = ndi.binary_opening((diff >= PATCH_D) & inner, disk(3))
    cl_map = cc_label(patch)
    keep = np.bincount(cl_map.ravel()) >= 0.02 * d * d
    keep[0] = False
    patch = keep[cl_map]
    regs = regions(face, u, v, fur)
    region_out = {}
    for name, r in regs.items():
        n = int(r.sum())
        if n < 30:
            region_out[name] = {"pixels": n, "visible": False}
            continue
        mean = lab[r].mean(0)
        pr = patch & r
        entry = {"pixels": n, "visible": True, "mean_lab": [round(float(x), 1) for x in mean],
                 "delta_e_main": round(delta_e(mean, main), 1), "patch_share": round(float(pr.sum() / n), 3)}
        if pr.sum() >= 10:
            entry["patch_lab"] = [round(float(x), 1) for x in lab[pr].mean(0)]
        region_out[name] = entry

    # 3 crown grid
    mid = ((face.eye_l + face.eye_r) / 2).numpy()
    right, up = face.right.numpy(), face.up.numpy()
    sub = 4
    gu = GRID_U[0] + (np.arange(GRID_N * sub) + 0.5) * (GRID_U[1] - GRID_U[0]) / (GRID_N * sub)
    gv = GRID_V[1] - (np.arange(GRID_N * sub) + 0.5) * (GRID_V[1] - GRID_V[0]) / (GRID_N * sub)
    UU, VV = np.meshgrid(gu, gv)
    X = mid[0] + d * (UU * right[0] + VV * up[0]) - 0.5
    Y = mid[1] + d * (UU * right[1] + VV * up[1]) - 0.5
    crown_s = map_coordinates(regs["crown"].astype(float), [Y, X], order=0, cval=0)
    patch_s = map_coordinates((patch & regs["crown"]).astype(float), [Y, X], order=0, cval=0)
    cs = crown_s.reshape(GRID_N, sub, GRID_N, sub).sum((1, 3))
    ps = patch_s.reshape(GRID_N, sub, GRID_N, sub).sum((1, 3))
    grid = np.where(cs > 0, ps / np.maximum(cs, 1), -1)

    # 4 eyes: point 0 is the left eye (picture left), point 1 the right
    eyes_lab = {}
    for key, c in (("eye_left", eyes[0]), ("eye_right", eyes[1])):
        disc = (dist(c) <= IRIS_R * d) & (m > 0.5)
        if disc.sum() < 5:
            eyes_lab[key] = None
            continue
        e = lab[disc]
        lo, hi = np.percentile(e[:, 0], [35, 95])
        e = e[(e[:, 0] >= lo) & (e[:, 0] <= hi)]
        eyes_lab[key] = [round(float(x), 1) for x in e.mean(0)]

    return {"id": rel, "stem": stem, "painted": painted, "box": list(box), "d_px": round(d, 1),
            "main_lab": [round(float(x), 1) for x in main],
            "palette": [{"lab": [round(float(x), 1) for x in c], "share": round(float(s), 3)}
                        for c, s in zip(centres, shares)],
            "pattern_features": {k: round(float(x), 3) for k, x in feat.items()},
            "regions": region_out, "crown_grid": np.round(grid, 2).tolist(), "eyes_lab": eyes_lab}


MIRROR_KEYS = [("eye_left", "eye_right"), ("ear_left", "ear_right"), ("cheek_left", "cheek_right"),
               ("eye_ring_left", "eye_ring_right")]


def mirror(row: dict) -> dict:
    """The description of the left-right mirrored photo, matching Brush GPT's MIRROR_POINTS: left and right eyes,
    ears and cheeks swap and the crown grid's columns reverse (a classified row; measures need classify first)."""
    out = json.loads(json.dumps(row))
    for a, b in MIRROR_KEYS:
        for d in (out, out["regions"], out["eyes_lab"]):
            if a in d and b in d:
                d[a], d[b] = d[b], d[a]
    out["crown_grid"] = [r[::-1] for r in out["crown_grid"]]
    return out


# ---------------------------------------------------------------- report

def font(size: int):
    from PIL import ImageFont
    for f in ("/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/STHeiti Medium.ttc"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def canvas_of(row: dict) -> np.ndarray:
    from sellanything_ml.sketch import catbrush
    photo = catbrush.load(str(AFHQ / row["id"]), str(MASKS / f"{row['stem']}_mask1.png"), tuple(row["box"]))
    return (photo.target.numpy().clip(0, 1) * 255).astype(np.uint8)


def mosaic(rows: list[dict], title: str, path: Path, caption, n: int = 12, tile: int = 160) -> None:
    from PIL import Image, ImageDraw
    pick = random.Random(0).sample(rows, min(n, len(rows)))
    cols = 6
    nrow = max(1, math.ceil(len(pick) / cols))
    W, head, cap = cols * tile, 34, 40
    im = Image.new("RGB", (W, head + nrow * (tile + cap)), "white")
    dr = ImageDraw.Draw(im)
    dr.text((8, 6), f"{title}  (共 {len(rows)} 隻,隨機 {len(pick)} 隻)", fill="black", font=font(18))
    for k, row in enumerate(pick):
        x, y = (k % cols) * tile, head + (k // cols) * (tile + cap)
        im.paste(Image.fromarray(canvas_of(row)).resize((tile, tile)), (x, y))
        for j, line in enumerate(caption(row)):
            dr.text((x + 3, y + tile + 2 + 13 * j), line, fill="black", font=font(11))
    im.save(path)


def eye_caption(r: dict) -> list[str]:
    return [r["stem"][-6:] + f"  ΔE {r['eye_delta_e']}",
            f"左眼 {ZH[r['eye_left']['name']]}  右眼 {ZH[r['eye_right']['name']]}(畫面左右)"]


def colour_caption(r: dict) -> list[str]:
    pal = " ".join(f"{ZH[p['name']]}{int(p['share'] * 100)}" for p in r["palette"])
    return [f"{r['stem'][-6:]} {ZH[r['main_category']]}/{ZH[r['pattern']]}", pal]


def region_caption(name: str):
    def cap(r: dict) -> list[str]:
        g = r["regions"][name]
        return [f"{r['stem'][-6:]} 主色 {ZH[r['base_colour']]}",
                f"斑 {ZH.get(g.get('patch_name', ''), '?')} {int(g['patch_share'] * 100)}%"]
    return cap


def crown_figure(rows: list[dict], path: Path, n: int = 16) -> None:
    """Photo with the crown region and its patch, next to the 16 × 16 grid drawn the same way up."""
    from PIL import Image, ImageDraw
    from sellanything_ml.sketch.oilprocess import Face
    labels = np.load(LABELS)
    ids = list(labels["ids"])
    pick = random.Random(1).sample(rows, min(n, len(rows)))
    tile, cols = 192, 2
    im = Image.new("RGB", (cols * (2 * tile + 30), 40 + math.ceil(len(pick) / cols) * (tile + 34)), "white")
    dr = ImageDraw.Draw(im)
    dr.text((8, 8), "頭頂斑塊:左 = 照片(綠框 = 16×16 格子的範圍),右 = 格子(白 = 斑塊佔比,灰 = 不是頭頂)",
            fill="black", font=font(16))
    for k, row in enumerate(pick):
        x0, y0 = (k % cols) * (2 * tile + 30), 40 + (k // cols) * (tile + 34)
        img = Image.fromarray(canvas_of(row))
        face = Face.from_photo(labels["points"][ids.index(row["id"])], tuple(row["box"]))
        mid = ((face.eye_l + face.eye_r) / 2).numpy()
        right, up = face.right.numpy(), face.up.numpy()
        corners = [(GRID_U[0], GRID_V[1]), (GRID_U[1], GRID_V[1]), (GRID_U[1], GRID_V[0]), (GRID_U[0], GRID_V[0])]
        poly = [tuple(mid + face.d * (cu * right + cv * up)) for cu, cv in corners]
        di = ImageDraw.Draw(img)
        di.polygon(poly, outline=(0, 200, 0))
        im.paste(img.resize((tile, tile)), (x0, y0))
        g = np.array(row["crown_grid"])
        rgb = np.where(g[..., None] < 0, [90, 90, 90], (g[..., None] * 255).clip(0, 255) * [1, 1, 1]).astype(np.uint8)
        im.paste(Image.fromarray(rgb).resize((tile, tile), Image.NEAREST), (x0 + tile + 6, y0))
        cr = row["regions"]["crown"]
        txt = f"{row['stem'][-6:]} 主色 {ZH[row['base_colour']]}  斑 {ZH.get(cr.get('patch_name', ''), '-')} " \
              f"{int(cr.get('patch_share', 0) * 100)}%"
        dr.text((x0, y0 + tile + 4), txt, fill="black", font=font(12))
    im.save(path)


REGION_RGB = {"crown": (230, 40, 40), "ear_left": (40, 200, 230), "ear_right": (40, 90, 230),
              "bridge": (240, 200, 0), "muzzle_chin": (200, 0, 200), "cheek_left": (150, 230, 80),
              "cheek_right": (0, 160, 60), "eye_ring_left": (255, 140, 0), "eye_ring_right": (255, 90, 160)}


def regions_figure(rows: list[dict], path: Path, n: int = 12) -> None:
    """The seven regions tinted over the photo, and the two eye discs, so left/right and the shapes can be checked."""
    from PIL import Image, ImageDraw
    from sellanything_ml.sketch.oilprocess import Face
    labels = np.load(LABELS)
    ids = list(labels["ids"])
    pick = random.Random(2).sample(rows, min(n, len(rows)))
    tile, cols = 256, 4
    im = Image.new("RGB", (cols * tile, 60 + math.ceil(len(pick) / cols) * (tile + 20)), "white")
    dr = ImageDraw.Draw(im)
    dr.text((8, 6), "區域(左右以畫面為準,與 cat-face-landmarks 的 NAMES 一致)。青圈 = 左眼,黃圈 = 右眼", fill="black",
            font=font(15))
    x = 8
    for k in REGIONS:
        dr.rectangle([x, 34, x + 14, 48], fill=REGION_RGB[k])
        dr.text((x + 18, 32), ZH[k], fill="black", font=font(14))
        x += 18 + 16 * len(ZH[k]) + 14
    for j, row in enumerate(pick):
        img = canvas_of(row).astype(float)
        face = Face.from_photo(labels["points"][ids.index(row["id"])], tuple(row["box"]))
        u, v = face_uv(face)
        fur = np.ones((SIDE, SIDE), bool)
        for name, r in regions(face, u, v, fur).items():
            img[r] = 0.55 * img[r] + 0.45 * np.array(REGION_RGB[name])
        pil = Image.fromarray(img.astype(np.uint8))
        di = ImageDraw.Draw(pil)
        for c, col in ((face.eye_l.numpy(), (0, 230, 230)), (face.eye_r.numpy(), (255, 220, 0))):
            r = IRIS_R * face.d
            di.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=col, width=2)
        x0, y0 = (j % cols) * tile, 60 + (j // cols) * (tile + 20)
        im.paste(pil, (x0, y0))
        dr.text((x0 + 3, y0 + tile + 2), row["stem"], fill="black", font=font(12))
    im.save(path)


def hand_check(rows: list[dict]) -> list[str]:
    """Agreement with runs/coat/handcheck/labels.txt (written by looking at the photos only): set A tuned the
    thresholds, set B did not. A label may list several acceptable answers when the photo is ambiguous."""
    ids_path, lab_path = OUT / "handcheck/ids.json", OUT / "handcheck/labels.txt"
    if not (ids_path.exists() and lab_path.exists()):
        return []
    by_id = {r["id"]: r for r in rows}
    want = {}
    for line in lab_path.read_text().splitlines():
        if line[:1] in ("A", "B"):
            key, colour, pattern = line.split()[:3]
            want[key] = (set(colour.split("|")), set(pattern.split("|")))
    out = ["\n## 與人工標註比對(A = 用來調門檻,B = 留出、只量一次)\n",
           "| 組 | 隻數 | 底色對 | 花紋對 |", "|---|---|---|---|"]
    confusion = Counter()
    for part in ("A", "B"):
        n = colour_ok = pattern_ok = 0
        for j, rid in enumerate(json.loads(ids_path.read_text())):
            key = ("A" if j < 48 else "B") + f"{j % 48 + 1:02d}"
            if key[0] != part or rid not in by_id:
                continue
            r, (cs, ps) = by_id[rid], want[key]
            n += 1
            colour_ok += r["base_colour"] in cs
            pattern_ok += r["pattern"] in ps
            if part == "B" and r["pattern"] not in ps:
                confusion[(sorted(ps)[0], r["pattern"])] += 1
        if n:
            out.append(f"| {part} | {n} | {colour_ok} ({colour_ok / n:.0%}) | {pattern_ok} ({pattern_ok / n:.0%}) |")
    if confusion:
        out.append("\nB 組花紋錯在哪(人工 → 演算法):" + "、".join(
            f"{ZH[a]} → {ZH[b]} ×{k}" for (a, b), k in confusion.most_common()))
    return out


def report() -> None:
    rows = [json.loads(x) for x in (OUT / "measures.jsonl").read_text().splitlines()]
    bad = [r for r in rows if "error" in r]
    rows = [classify(r) for r in rows if "error" not in r]
    with open(OUT / "coat.jsonl", "w") as f:
        for r in rows + bad:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    painted = [r for r in rows if r["painted"]]
    lines = ["# 貓的外觀描述:統計\n", f"處理 {len(rows) + len(bad)} 隻,失敗 {len(bad)} 隻;"
             f"其中有 Brush GPT 畫作(訓練資料)的 {len(painted)} 隻。括號內是畫過的那一批。\n"]

    def table(title: str, key, values: list[str]) -> None:
        c, cp = Counter(key(r) for r in rows), Counter(key(r) for r in painted)
        lines.append(f"\n## {title}\n\n| 類別 | 全部 | (有畫作) |\n|---|---|---|")
        for v in values:
            lines.append(f"| {ZH.get(v, v)} `{v}` | {c[v]} | ({cp[v]}) |")

    table("主色類別", lambda r: r["main_category"], FUR_NAMES + ["mixed"])
    table("底色(最多的毛色,混色貓也有)", lambda r: r["base_colour"], FUR_NAMES)
    table("花紋", lambda r: r["pattern"], PATTERNS)
    table("左眼(畫面左)", lambda r: r["eye_left"]["name"], EYE_NAMES)
    table("右眼(畫面右)", lambda r: r["eye_right"]["name"], EYE_NAMES)
    table("異色瞳", lambda r: str(r["odd_eyes"]), ["True", "False"])
    lines.append("\n## 各區域有斑塊(與主色差 ≥ 門檻、佔該區 ≥ 20%)\n\n| 區域 | 看得到 | 有斑塊 | (有畫作) | 斑塊顏色 |\n|---|---|---|---|---|")
    for name in REGIONS:
        vis = [r for r in rows if r["regions"][name]["visible"]]
        has = [r for r in vis if r["regions"][name]["has_patch"]]
        hp = [r for r in has if r["painted"]]
        col = Counter(r["regions"][name].get("patch_name") for r in has)
        lines.append(f"| {ZH[name]} `{name}` | {len(vis)} | {len(has)} | ({len(hp)}) | "
                     + ", ".join(f"{ZH.get(k, k)} {v}" for k, v in col.most_common()) + " |")
    lines += hand_check(rows)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    figs = OUT / "mosaics"
    figs.mkdir(exist_ok=True)
    for v in FUR_NAMES + ["mixed"]:
        sel = [r for r in rows if r["main_category"] == v]
        if sel:
            mosaic(sel, f"主色 {ZH[v]} ({v})", figs / f"colour_{v}.png", colour_caption)
    for v in PATTERNS:
        sel = [r for r in rows if r["pattern"] == v]
        if sel:
            mosaic(sel, f"花紋 {ZH[v]} ({v})", figs / f"pattern_{v}.png", colour_caption)
    for v in EYE_NAMES:
        sel = [r for r in rows if v in (r["eye_left"]["name"], r["eye_right"]["name"])]
        if sel:
            mosaic(sel, f"至少一眼 {ZH[v]} ({v})", figs / f"eye_{v}.png", eye_caption)
    sel = [r for r in rows if r["odd_eyes"]]
    if sel:
        mosaic(sel, "異色瞳", figs / "eye_odd.png", eye_caption, n=min(24, len(sel)))
    for name in REGIONS:
        sel = [r for r in rows if r["regions"][name].get("has_patch")]
        if sel:
            mosaic(sel, f"{ZH[name]} 有斑塊", figs / f"patch_{name}.png", region_caption(name))
    regions_figure(rows, OUT / "regions_overlay.png")
    crown = [r for r in rows if r["regions"]["crown"].get("has_patch")]
    if crown:
        crown_figure(crown, OUT / "crown_grid_vs_photo.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="only the first N cats (in a fixed random order)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--report", action="store_true",
                    help="only classify runs/coat/measures.jsonl again and redraw the table and mosaics")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.report:
        n = len(np.load(LABELS)["ids"])
        order = list(range(n))
        random.Random(0).shuffle(order)
        if args.limit:
            order = order[:args.limit]
        with Pool(args.workers) as pool, open(OUT / "measures.jsonl", "w") as f:
            for k, row in enumerate(pool.imap(describe, order, chunksize=8)):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if (k + 1) % 250 == 0:
                    print(f"  {k + 1}/{len(order)}", flush=True)
    report()


if __name__ == "__main__":
    main()
