"""The market stage, the girl and the UI font, built from the source art in art/stage/.

    python scripts/make_stage_art.py          # writes game/assets/stage/ and game/assets/fonts/

Sources: background_tile_crossfade.png (the owner's watercolor strip, its ends cross-faded so it repeats
without a seam), stall_cloth.png (the stall with its hanging cloth, on black), girl_idle.gif (12 front-facing
frames on cream), cat_shelter.png (the shelter, on transparent, with eight empty easels), girl_run.gif (49 frames on transparent: front, starting to run, running right, stopping),
girl_walk.gif (30 frames: walking away from the camera, turning, walking towards it),
npc_<n>.gif (a buyer: walking away from the camera, turning, walking towards it), npc_<n>_idle.png
(the same buyer standing front-facing with both feet planted), Nunito.ttf (SIL OFL,
google/fonts). Where things stand is in game/assets/stage/layout.json, written here
only when it does not exist yet, so it can be tuned by hand.
Sheets leave a transparent border round every cell, so smooth sampling never bleeds from a neighbour.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "art" / "stage"
STAGE = ROOT / "game" / "assets" / "stage"
FONTS = ROOT / "game" / "assets" / "fonts"
SCREEN_H = 720
PAD = 2                                        # transparent border round each sheet cell
SHEET_COLS = 16                                # keep every OpenGL texture safely below 16,384 px wide
FONT_SIZES = (14, 18, 22, 28, 36, 48)          # atlas sizes; text is drawn from the nearest larger one
ROUND_PX, ROUND_SUPER, ROUND_RADIUS, ROUND_EDGE = 64, 4, 18, 10   # rounded texture: size, supersampling, corner, edge
STALL_PX = 640                                 # stall texture height (drawn at layout.json's height)
CLOTH = (420, 340, 1130, 830)                  # where one can draw on the cloth, in stall_cloth.png pixels
GIRL_PX = 320                                  # girl texture height when standing (drawn at layout.json's)
RUN_PARTS = {"start": (1, 6), "run": (7, 41), "stop": (42, 48)}   # girl_run.gif frames, inclusive
NPC_PARTS = {"away": (0, 11), "turn": (12, 17), "toward": (18, 29)}   # npc_<n>.gif and girl_walk.gif, inclusive
JUMP_FRAMES = {"rise": 7, "top": 10, "fall": 25, "land": 34}         # girl_run.gif frames shown while jumping
RUNNING_SHARE = 0.85   # girl_run.gif draws her as tall running as standing, so her head grows; running she
                       # is drawn this share of her standing height, easing in and out while she starts and stops
SHELTER_PX = 880                               # shelter texture height: about the source's own
# The blank canvases on the shelter's easels, in cat_shelter.png pixels: where adoptable cats will show.
EASELS = [(282, 498, 381, 604), (433, 498, 532, 604), (580, 498, 682, 604), (737, 498, 839, 604),
          (938, 498, 1039, 604), (1097, 498, 1199, 604), (1246, 498, 1344, 604), (1395, 498, 1496, 604)]
# World x is 0 at the stall's centre; "bottom" is the screen y the object stands on, "height" its drawn height.
LAYOUT = {"objects": [{"image": "stall", "x": 0, "bottom": 500, "height": 470},
                      {"image": "shelter", "x": 1500, "bottom": 520, "height": 420}],
          "ground": 545, "girl_height": 200, "girl_start": 330, "buyer_scale": 2.25, "npc_height": 190}


def keyed(frame: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    """The figure without its flat background, including background trapped between the character's feet."""
    rgb = np.array(frame.convert("RGB")).astype(float)
    near = np.sqrt(((rgb - np.array(color, float)) ** 2).sum(-1)) < 22
    lab, _ = ndimage.label(near)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    remove = np.isin(lab, list(border))
    # The idle source's boots enclose a narrow column of the same flat backdrop, so flood-fill from the border
    # cannot reach it. Remove only tall, narrow backdrop-colour components low and near the frame centre; this
    # selects the gap in all 12 frames without touching cream highlights on her face, hair, blouse or skirt.
    h, w = near.shape
    for component in range(1, int(lab.max()) + 1):
        if component in border:
            continue
        ys, xs = np.nonzero(lab == component)
        if not len(xs):
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        centred = abs((x0 + x1) / 2 - w / 2) < w * 0.08
        foot_gap = centred and y0 > h * 0.68 and y1 - y0 > h * 0.04 and x1 - x0 < w * 0.08
        if foot_gap:
            remove |= lab == component
    alpha = ndimage.gaussian_filter(np.where(remove, 0.0, 1.0), 0.8)
    return Image.fromarray(np.dstack([rgb.astype(np.uint8), (alpha * 255).astype(np.uint8)]), "RGBA")


def soft_edge(frame: Image.Image) -> Image.Image:
    """A GIF frame's all-or-nothing transparency, softened by a fraction of a pixel."""
    a = np.array(frame)
    alpha = (ndimage.gaussian_filter(a[..., 3] / 255.0, 0.6) * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([a[..., :3], alpha]), "RGBA")


def gif_frames(path: Path) -> tuple[list[Image.Image], list[int]]:
    im = Image.open(path)
    out, ms = [], []
    for i in range(im.n_frames):
        im.seek(i)
        out.append(im.convert("RGBA"))
        ms.append(int(im.info.get("duration", 100)))
    return out, ms


def sheet(cells: list[Image.Image], name: str, ms: list[int], anchor: tuple[int, int], **extra: object) -> None:
    """A row-major grid of equal cells plus metadata; cells keep their PAD border on every side."""
    cw, ch = cells[0].size
    cols = min(SHEET_COLS, len(cells))
    rows = (len(cells) + cols - 1) // cols
    cell_w, cell_h = cw + 2 * PAD, ch + 2 * PAD
    out = Image.new("RGBA", (cell_w * cols, cell_h * rows))
    for i, c in enumerate(cells):
        out.paste(c, ((i % cols) * cell_w + PAD, (i // cols) * cell_h + PAD))
    out.save(STAGE / f"{name}.png", optimize=True)
    meta = {"cell": [cell_w, cell_h], "anchor": [anchor[0] + PAD, anchor[1] + PAD], "cols": cols,
            "rows": rows, "frames": len(cells), "frame_ms": ms, **extra}
    (STAGE / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n")


def figure_sheet(frames: list[Image.Image], name: str, ms: list[int], standing_height: int, **extra: object) -> None:
    """All frames cut by one box (their union), scaled so the first frame's figure is standing_height tall;
    the feet are the bottom centre of the first frame's figure."""
    box = Image.new("L", frames[0].size)
    for f in frames:
        box.paste(255, mask=f.getchannel("A"))
    left, top, right, bottom = box.getbbox()
    fl, ft, fr, fb = frames[0].getchannel("A").getbbox()
    k = standing_height / (fb - ft)
    size = (round((right - left) * k), round((bottom - top) * k))
    cells = [f.crop((left, top, right, bottom)).resize(size, Image.LANCZOS) for f in frames]
    sheet(cells, name, ms, (round(((fl + fr) / 2 - left) * k), round((fb - top) * k)), standing=standing_height,
          **extra)


def fitted_sheet(frames: list[Image.Image], name: str, ms: list[int], heights: list[float], **extra: object) -> None:
    """Each frame scaled on its own so its figure is heights[i] tall, its feet (the middle of its lowest fifth)
    at one anchor: for drawings whose figure changes size from frame to frame."""
    fitted, feet_x = [], []
    for f, h in zip(frames, heights, strict=True):
        im = f.crop(f.getchannel("A").getbbox())
        im = im.resize((round(im.width * h / im.height), round(h)), Image.LANCZOS)
        low = np.array(im.getchannel("A"))[int(im.height * 0.8):].astype(float)
        feet_x.append(float((low.sum(0) * np.arange(low.shape[1])).sum() / max(low.sum(), 1.0)))
        fitted.append(im)
    left = max(fx for fx in feet_x)
    right = max(im.width - fx for im, fx in zip(fitted, feet_x, strict=True))
    cell_h = round(max(heights))
    cells = []
    for im, fx in zip(fitted, feet_x, strict=True):
        cell = Image.new("RGBA", (round(left + right), cell_h))
        cell.paste(im, (round(left - fx), cell_h - im.height))
        cells.append(cell)
    sheet(cells, name, ms, (round(left), cell_h), standing=GIRL_PX, **extra)


def girl() -> None:
    idle, idle_ms = gif_frames(SRC / "girl_idle.gif")
    idle = [keyed(f, idle[0].convert("RGB").getpixel((1, 1))) for f in idle]
    run, run_ms = gif_frames(SRC / "girl_run.gif")
    run = [soft_edge(f) for f in run]
    figure_sheet(idle, "girl_idle", idle_ms, GIRL_PX)
    shares = []
    for i in range(len(run)):
        (s0, s1), (t0, t1) = RUN_PARTS["start"], RUN_PARTS["stop"]
        if i < s0 or i > t1:
            shares.append(1.0)
        elif i <= s1:
            shares.append(1.0 + (RUNNING_SHARE - 1.0) * (i - s0 + 1) / (s1 - s0 + 2))
        elif i < t0:
            shares.append(RUNNING_SHARE)
        else:
            shares.append(RUNNING_SHARE + (1.0 - RUNNING_SHARE) * (i - t0 + 1) / (t1 - t0 + 2))
    fitted_sheet(run, "girl_run", run_ms, [GIRL_PX * k for k in shares], parts=RUN_PARTS, jump=JUMP_FRAMES)
    walk, walk_ms = gif_frames(SRC / "girl_walk.gif")
    figure_sheet([soft_edge(f) for f in walk], "girl_walk", walk_ms, GIRL_PX, parts=NPC_PARTS)


def npcs() -> None:
    """Build each buyer's walking sheet plus a separate, genuinely planted front-facing idle pose."""
    for path in sorted(SRC.glob("npc_*.gif")):
        frames, ms = gif_frames(path)
        frames = [soft_edge(f) for f in frames]
        figure_sheet(frames, path.stem, ms, GIRL_PX, parts=NPC_PARTS)
        idle_path = path.with_name(f"{path.stem}_idle.png")
        if not idle_path.exists():
            raise FileNotFoundError(f"missing idle art for {path.stem}: {idle_path}")
        idle = Image.open(idle_path).convert("RGBA")
        figure_sheet([idle], f"{path.stem}_idle", [1000], GIRL_PX)


def stall() -> None:
    src = np.array(Image.open(SRC / "stall_cloth.png").convert("RGB")).astype(float)
    alpha = np.clip((src.max(-1) - 12) / 60, 0, 1)            # the black background, faded at soft edges
    rgb = np.clip(src / np.maximum(alpha, 1e-3)[..., None], 0, 255)
    im = Image.fromarray(np.dstack([rgb.astype(np.uint8), (alpha * 255).astype(np.uint8)]), "RGBA")
    left, top, right, bottom = im.getchannel("A").getbbox()
    k = STALL_PX / (bottom - top)
    im = im.crop((left, top, right, bottom)).resize((round((right - left) * k), STALL_PX), Image.LANCZOS)
    im.save(STAGE / "stall.png", optimize=True)
    cloth = [round((CLOTH[0] - left) * k), round((CLOTH[1] - top) * k), round((CLOTH[2] - left) * k),
             round((CLOTH[3] - top) * k)]
    (STAGE / "stall.json").write_text(json.dumps({"size": list(im.size), "cloth": cloth}, indent=1) + "\n")


def shelter() -> None:
    im = Image.open(SRC / "cat_shelter.png").convert("RGBA")
    left, top, right, bottom = im.getchannel("A").getbbox()
    k = SHELTER_PX / (bottom - top)
    im = im.crop((left, top, right, bottom)).resize((round((right - left) * k), SHELTER_PX), Image.LANCZOS)
    im.save(STAGE / "shelter.png", optimize=True)
    easels = [[round((x0 - left) * k), round((y0 - top) * k), round((x1 - left) * k), round((y1 - top) * k)]
              for x0, y0, x1, y1 in EASELS]
    (STAGE / "shelter.json").write_text(json.dumps({"size": list(im.size), "easels": easels}, indent=1) + "\n")


def background() -> None:
    bg = Image.open(SRC / "background_tile_crossfade.png").convert("RGB")
    bg = bg.resize((round(bg.width * SCREEN_H / bg.height), SCREEN_H), Image.LANCZOS)
    bg.save(STAGE / "background.png", optimize=True)


def rounded() -> None:
    """White rounded-rectangle fill and edge, ROUND_PX square, for nine-slice panels and buttons. Drawn
    ROUND_SUPER times larger and shrunk, so the curves are smooth."""
    big = ROUND_SUPER * ROUND_PX
    inset = 2
    for name, width in (("round_fill", 0), ("round_edge", ROUND_EDGE)):
        im = Image.new("L", (big, big))
        d = ImageDraw.Draw(im)
        box = (inset, inset, big - 1 - inset, big - 1 - inset)
        if width:
            d.rounded_rectangle(box, ROUND_SUPER * ROUND_RADIUS, outline=255, width=width)
        else:
            d.rounded_rectangle(box, ROUND_SUPER * ROUND_RADIUS, fill=255)
        out = Image.new("RGBA", (ROUND_PX, ROUND_PX), (255, 255, 255, 0))
        out.putalpha(im.resize((ROUND_PX, ROUND_PX), Image.LANCZOS))
        out.save(STAGE / f"{name}.png", optimize=True)


def font() -> None:
    """Nunito Bold, printable ASCII, at each of FONT_SIZES, white on transparent, in one atlas.

    nunito.txt: per size a line "size <px> <ascent> <line height>", then one line per glyph:
    "<code> <x> <y> <w> <h> <left> <top> <advance>", where left/top place the glyph image relative to the
    pen position on the baseline.
    """
    lines, placed, x, y, row = [], [], 0, 0, 0
    atlas_w = 1024
    for px in FONT_SIZES:
        f = ImageFont.truetype(str(SRC / "Nunito.ttf"), px)
        f.set_variation_by_name("Bold")
        ascent, descent = f.getmetrics()
        lines.append(f"size {px} {ascent} {ascent + descent}")
        for code in range(32, 127):
            ch = chr(code)
            gl, gt, gr, gb = f.getbbox(ch, anchor="ls")
            w, h = max(gr - gl, 0) + 2, max(gb - gt, 0) + 2
            if x + w + 2 > atlas_w:
                x, y, row = 0, y + row + 2, 0
            placed.append((f, ch, x + 1, y + 1, gl, gt))
            lines.append(f"{code} {x} {y} {w} {h} {gl - 1} {gt - 1} {f.getlength(ch):.2f}")
            x, row = x + w + 2, max(row, h)
        x, y, row = 0, y + row + 2, 0
    atlas = Image.new("L", (atlas_w, y))
    d = ImageDraw.Draw(atlas)
    for f, ch, gx, gy, gl, gt in placed:
        d.text((gx - gl, gy - gt), ch, font=f, fill=255, anchor="ls")
    out = Image.new("RGBA", atlas.size, (255, 255, 255, 0))
    out.putalpha(atlas)
    out.save(FONTS / "nunito.png", optimize=True)
    (FONTS / "nunito.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Build every stage asset and the font, and list what was written."""
    STAGE.mkdir(parents=True, exist_ok=True)
    FONTS.mkdir(parents=True, exist_ok=True)
    background()
    stall()
    shelter()
    girl()
    npcs()
    rounded()
    font()
    layout = STAGE / "layout.json"
    if not layout.exists():
        layout.write_text(json.dumps(LAYOUT, indent=1) + "\n")
    for p in sorted([*STAGE.iterdir(), *FONTS.iterdir()]):
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
