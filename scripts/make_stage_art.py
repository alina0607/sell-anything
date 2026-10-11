"""The market stage, the girl and the UI font, built from the source art in art/stage/.

    python scripts/make_stage_art.py          # writes game/assets/stage/ and game/assets/fonts/

Sources: background_tile_crossfade.png (the owner's watercolor strip, its ends cross-faded so it repeats
without a seam), stall_cloth.png (the stall with its hanging cloth, on black), girl_idle.gif (12 front-facing
frames on cream), cat_shelter.png (the shelter, on transparent, with eight empty easels), girl_run.gif
(front, starting and stopping poses), girl_run_cycle_v3.png (four equally timed opposing gait phases),
girl_walk.gif (30 frames: walking away from the camera, turning, walking towards it),
npc_<n>.gif (a buyer: walking away from the camera, turning, walking towards it), npc_<n>_idle.png
(the same buyer standing back-facing with both feet planted), Nunito.ttf (SIL OFL,
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
RUN_PARTS = {"start": (1, 6), "run": (7, 10), "stop": (11, 17)}  # rebuilt output sheet, inclusive
NPC_PARTS = {"away": (0, 11), "turn": (12, 17), "toward": (18, 29)}   # npc_<n>.gif and girl_walk.gif, inclusive
JUMP_FRAMES = {"rise": 7, "top": 8, "fall": 9, "land": 17}
SHELTER_PX = 880                               # shelter texture height: about the source's own
# The blank canvases on the shelter's easels, in cat_shelter.png pixels: where adoptable cats will show.
EASELS = [(282, 498, 381, 604), (433, 498, 532, 604), (580, 498, 682, 604), (737, 498, 839, 604),
          (938, 498, 1039, 604), (1097, 498, 1199, 604), (1246, 498, 1344, 604), (1395, 498, 1496, 604)]
# World x is 0 at the stall's centre; "bottom" is the screen y the object stands on, "height" its drawn height.
LAYOUT = {"objects": [{"image": "stall", "x": 0, "bottom": 500, "height": 470},
                      {"image": "shelter", "x": 1500, "bottom": 520, "height": 420}],
          "ground": 545, "girl_height": 200, "girl_start": 330, "buyer_scale": 3.125, "npc_height": 200}


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


def grade_girl_idle(frame: Image.Image) -> Image.Image:
    """Match the pale idle source to the deeper, more saturated walking art without redrawing it."""
    a = np.array(frame).astype(float)
    rgb = a[..., :3]
    gray = rgb.mean(-1, keepdims=True)
    a[..., :3] = np.clip((gray + (rgb - gray) * 1.20) * 0.943, 0, 255)
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def grade_to(frame: Image.Image, target_value: float = 170.0, target_saturation: float = 74.0) -> Image.Image:
    """Give every motion frame one measured value/saturation target instead of eyeballing sources."""
    a = np.array(frame).astype(float)
    mask = a[..., 3] > 20
    rgb = a[..., :3]
    gray = rgb.mean(-1, keepdims=True)
    saturation = (rgb[mask].max(1) - rgb[mask].min(1)).mean()
    rgb = gray + (rgb - gray) * (target_saturation / max(saturation, 1.0))
    value = rgb[mask].mean(-1).mean()
    a[..., :3] = np.clip(rgb * (target_value / max(value, 1.0)), 0, 255)
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def gif_frames(path: Path) -> tuple[list[Image.Image], list[int]]:
    im = Image.open(path)
    out, ms = [], []
    for i in range(im.n_frames):
        im.seek(i)
        out.append(im.convert("RGBA"))
        ms.append(int(im.info.get("duration", 100)))
    return out, ms


def main_component(frame: Image.Image) -> Image.Image:
    """Remove disconnected transparent-edge paint flecks from one ImageGen sprite."""
    frame = frame.copy().convert("RGBA")
    alpha = np.array(frame.getchannel("A"))
    labelled, count = ndimage.label(alpha > 20)
    if count:
        sizes = ndimage.sum(alpha, labelled, range(1, count + 1))
        keep = labelled == int(np.argmax(sizes) + 1)
        frame.putalpha(Image.fromarray(np.where(keep, alpha, 0).astype(np.uint8), "L"))
    return frame


def visible_bbox(frame: Image.Image, threshold: int = 20) -> tuple[int, int, int, int]:
    """Bounding box of paint a player can actually see, excluding near-transparent ImageGen residue."""
    alpha = frame.getchannel("A").point(lambda value: 255 if value > threshold else 0)
    box = alpha.getbbox()
    if box is None:
        raise ValueError("sprite frame has no visible pixels")
    return box


def grid_frames(path: Path, cols: int, rows: int) -> list[Image.Image]:
    """Split an ImageGen sheet and keep only each cell's main connected alpha component."""
    im = Image.open(path).convert("RGBA")
    frames = []
    for row in range(rows):
        for col in range(cols):
            x0, x1 = round(col * im.width / cols), round((col + 1) * im.width / cols)
            y0, y1 = round(row * im.height / rows), round((row + 1) * im.height / rows)
            frames.append(main_component(im.crop((x0, y0, x1, y1))))
    return frames


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
        visible = f.getchannel("A").point(lambda value: 255 if value > 20 else 0)
        box.paste(255, mask=visible)
    left, top, right, bottom = box.getbbox()
    fl, ft, fr, fb = visible_bbox(frames[0])
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


def scaled_sheet(frames: list[Image.Image], name: str, ms: list[int], scales: list[float], **extra: object) -> None:
    """Place frames at explicit scales, aligned by their feet.

    Unlike fitted_sheet, this never enlarges a crouched/crossing pose to fill the same total height. That
    old fitting made the head pulse larger whenever the legs moved closer together.
    """
    fitted, feet_x = [], []
    for frame, scale in zip(frames, scales, strict=True):
        im = frame.crop(frame.getchannel("A").getbbox())
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
        low = np.array(im.getchannel("A"))[round(im.height * 0.8):].astype(float)
        feet_x.append(float((low.sum(0) * np.arange(low.shape[1])).sum() / max(low.sum(), 1.0)))
        fitted.append(im)
    left = max(feet_x)
    right = max(im.width - fx for im, fx in zip(fitted, feet_x, strict=True))
    cell_h = max(GIRL_PX, max(im.height for im in fitted))
    cells = []
    for im, fx in zip(fitted, feet_x, strict=True):
        cell = Image.new("RGBA", (round(left + right), cell_h))
        cell.paste(im, (round(left - fx), cell_h - im.height))
        cells.append(cell)
    sheet(cells, name, ms, (round(left), cell_h), standing=GIRL_PX, **extra)


def girl() -> None:
    idle, idle_ms = gif_frames(SRC / "girl_idle.gif")
    idle = [grade_girl_idle(keyed(f, idle[0].convert("RGB").getpixel((1, 1)))) for f in idle]
    old_run, old_ms = gif_frames(SRC / "girl_run.gif")
    cycle = grid_frames(SRC / "girl_run_cycle_v3.png", 4, 1)
    # Four equal phases: stride A, crossing A, the truly opposing stride B, crossing B. The source sheet
    # has one locked head size and palette; each phase lasts exactly 0.5 seconds.
    run = [*old_run[:7], *cycle, *old_run[42:]]
    run_ms = [*old_ms[:7], *([500] * len(cycle)), *old_ms[42:]]
    run = [grade_to(soft_edge(f)) for f in run]
    figure_sheet(idle, "girl_idle", idle_ms, GIRL_PX)
    transitions = [*old_run[:7], *old_run[42:]]
    transition_scales = [GIRL_PX / (f.getchannel("A").getbbox()[3] - f.getchannel("A").getbbox()[1])
                         for f in transitions]
    cycle_scale = 175.0 / 290.0  # locked source head height -> the 175 px head in idle/walk sheets
    scales = [*transition_scales[:7], *([cycle_scale] * 4), *transition_scales[7:]]
    scaled_sheet(run, "girl_run", run_ms, scales, parts=RUN_PARTS, jump=JUMP_FRAMES)
    walk, walk_ms = gif_frames(SRC / "girl_walk.gif")
    figure_sheet([soft_edge(f) for f in walk], "girl_walk", walk_ms, GIRL_PX, parts=NPC_PARTS)


def npcs() -> None:
    """Build each buyer's walking sheet plus a separate, genuinely planted back-facing idle pose."""
    for path in sorted(SRC.glob("npc_*.gif")):
        frames, ms = gif_frames(path)
        frames = [main_component(soft_edge(f)) for f in frames]
        figure_sheet(frames, path.stem, ms, GIRL_PX, parts=NPC_PARTS)
        idle_path = path.with_name(f"{path.stem}_idle.png")
        if not idle_path.exists():
            raise FileNotFoundError(f"missing idle art for {path.stem}: {idle_path}")
        idle = main_component(Image.open(idle_path).convert("RGBA"))
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
    # Multiply the paintings into the stall cloth's watercolor texture. White becomes fabric instead of an
    # artificial white square, while every painted cat stroke keeps its original color and contrast.
    cat_out = STAGE / "shelter_cats"
    cat_out.mkdir(exist_ok=True)
    cloth = Image.open(SRC / "stall_cloth.png").convert("RGB").crop((510, 400, 1040, 760))
    for cat_path in sorted((SRC / "shelter_cats").glob("cat_*.png")):
        cat = Image.open(cat_path).convert("RGB")
        texture = cloth.resize(cat.size, Image.LANCZOS)
        painted = np.asarray(cat, dtype=np.uint16) * np.asarray(texture, dtype=np.uint16) // 255
        Image.fromarray(painted.astype(np.uint8), "RGB").save(cat_out / cat_path.name, optimize=True)


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
