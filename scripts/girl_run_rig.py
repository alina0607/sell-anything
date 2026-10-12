"""The girl's 8-frame run cycle, rebuilt as a cut-out puppet from one drawing (art/stage/girl_run_pose_b.png).
女孩的 8 格跑步循環：把一張原畫（girl_run_pose_b.png）拆成紙娃娃（cut-out）圖層後重新組出來。

Why a puppet / 為什麼用紙娃娃:
    The source sheet only ever drew her near (right) arm forward, so half of the old cycle swung the same arm and
    leg forward together. Here every part is a layer turned about its joint, so arm and leg always move in
    opposition: near arm forward <-> near leg back, and the reverse half-stride.
    原本的圖只畫過「近側（右）手在前」，所以舊循環有一半是同手同腳。這裡每個部位是一個圖層、繞關節旋轉，
    手腳永遠對側擺動：近側手在前 <-> 近側腳在後，下半步則相反。

    python scripts/girl_run_rig.py      # writes docs/images/girl_run_cycle.gif / 輸出預覽 GIF

scripts/make_stage_art.py calls cycle() to put these frames in the game's girl_run sheet.
scripts/make_stage_art.py 會呼叫 cycle()，把這些格子放進遊戲的 girl_run 圖集。

Coordinates are pixels of girl_run_pose_b.png (1358 x 1158). / 座標都是 girl_run_pose_b.png 的像素。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "art" / "stage" / "girl_run_pose_b.png"
PREVIEW = ROOT / "docs" / "images" / "girl_run_cycle.gif"
PAD = 260                                  # room round the drawing for limbs that swing out / 四周留白給擺出去的手腳

# --- Joints and cut lines, in source pixels / 關節與切線（原圖像素） ---
SHOULDER = (848, 612)                      # near shoulder, inside the puffy sleeve / 近側肩膀（在泡泡袖裡）
ELBOW = (905, 696)                         # where the forearm leaves the sleeve / 前臂從袖口出來的地方
FAR_SHOULDER = (858, 606)                  # far shoulder: just behind the near one / 遠側肩膀：在近側肩膀正後方
FRONT_HIP = (905, 885)                     # hip of the leg drawn in front / 畫在前方那條腿的髖關節
BACK_HIP = (730, 885)                      # hip of the leg drawn behind / 畫在後方那條腿的髖關節
# The forearm and fist: a loose box, refined by colour / 前臂與拳頭：粗框，再用顏色細修
ARM_BOX = [(888, 655), (940, 630), (1030, 602), (1068, 566), (1122, 572), (1140, 640), (1112, 694),
           (1000, 728), (920, 738), (888, 722)]
# The sleeve stays on the body and is redrawn over the arm, so the arm seems to come out of it.
# 袖子留在身體上，並蓋在手臂上面重畫一次，看起來手臂就是從袖子裡伸出來的。
SLEEVE = [(784, 560), (830, 532), (884, 544), (908, 600), (906, 700), (862, 708), (800, 694), (778, 640)]
# What lay behind the forearm: the chest (opaque) and her hair above the arm's upper edge.
# 前臂後面原本被擋住的東西：胸口（不透明）以及手臂上緣以上的頭髮。
CHEST = [(880, 620), (958, 620), (955, 700), (962, 742), (880, 745)]
HAIR = [(940, 540), (1170, 520), (1160, 538), (1130, 556), (1090, 573), (1050, 588), (1000, 600), (958, 610),
        (940, 610)]
# Where the near leg's white sock goes when that leg is drawn in front / 近側腿換到前面時，白襪要畫的位置
SOCK = [(905, 948), (1030, 940), (1048, 1000), (918, 1006)]
SOCK_WHITE = np.array([250, 246, 240], float)
# The bottom of the skirt's ruffle: legs are whatever lies below it.
# 裙擺荷葉邊的下緣：線以下的東西才算腿。
HEM = [(380, 860), (600, 872), (625, 900), (650, 922), (700, 945), (750, 958), (800, 956), (850, 952),
       (900, 942), (950, 925), (1000, 905), (1050, 878), (1200, 860)]
FRONT_LEG = [(850, 880), (1060, 880), (1190, 980), (1190, 1158), (900, 1158), (850, 1000)]
BACK_LEG = [(390, 850), (720, 870), (740, 1010), (600, 1010), (590, 1130), (390, 1130)]
SKIN_NEAR = np.array([252, 207, 186], float)   # lit skin of the near leg / 近側腿的亮面膚色
SKIN_FAR = np.array([205, 152, 139], float)    # shaded skin of the far leg / 遠側腿的暗面膚色
OUTLINE = (150, 82, 66, 255)                   # the drawing's brown line colour / 原畫的咖啡色線條

# One stride, frame by frame / 一步一步的格子設定:
#   near_front: is the near (right) leg the one in front? / 近側（右）腳是不是在前面
#   front, back: turn of the front / back leg about its hip, degrees (+ = counter-clockwise on screen)
#                前腿 / 後腿繞髖關節轉的角度（正 = 螢幕上逆時針）
#   body: how far the upper body sinks onto the legs / 上半身往下沉多少
#   lift: how far the feet are off the ground (flight) / 腳離地多高（騰空）
# Contact -> down -> passing -> flight, then the same with the legs swapped.
# 著地 -> 下沉 -> 交錯 -> 騰空，然後換另一隻腳再一次。
FRAMES = [
    dict(near_front=False, front=0, back=0, body=4, lift=0),        # contact, right leg back / 著地，右腳在後
    dict(near_front=False, front=-11, back=14, body=14, lift=0),    # down / 下沉
    dict(near_front=False, front=-22, back=34, body=8, lift=0),     # passing / 交錯
    dict(near_front=True, front=-9, back=12, body=-4, lift=22),     # flight / 騰空
    dict(near_front=True, front=0, back=0, body=4, lift=0),         # contact, right leg forward / 著地，右腳在前
    dict(near_front=True, front=-11, back=14, body=14, lift=0),
    dict(near_front=True, front=-22, back=34, body=8, lift=0),
    dict(near_front=False, front=-9, back=12, body=-4, lift=22),
]
ARM_SWING = -112         # how far the arm turns from fully forward to fully back / 手臂從最前擺到最後的角度


def polygon(points: list[tuple[int, int]], size: tuple[int, int]) -> np.ndarray:
    """A filled polygon as a boolean mask. / 把多邊形填成布林遮罩。"""
    m = Image.new("L", size)
    ImageDraw.Draw(m).polygon(points, fill=255)
    return np.array(m) > 0


def below(points: list[tuple[int, int]], size: tuple[int, int]) -> np.ndarray:
    """True under a polyline (y grows downward). / 折線以下為 True（y 往下變大）。"""
    xs, ys = zip(*points, strict=True)
    line = np.interp(np.arange(size[0]), xs, ys)
    return np.arange(size[1])[:, None] > line[None, :]


def layer(rgba: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A copy of the drawing with only the masked pixels kept. / 只保留遮罩內像素的圖層。"""
    out = rgba.copy()
    out[..., 3] = np.where(mask, rgba[..., 3], 0)
    return out


def capsule(size: tuple[int, int], a: tuple[float, float], b: tuple[float, float], width: int,
            fill: tuple[int, int, int]) -> np.ndarray:
    """A rounded limb from a to b with an outline: the hidden top of a limb, so turning it never opens a gap.
    畫一段有外框的圓頭肢體：補上被衣服遮住的那一截，旋轉時才不會露出縫。"""
    im = Image.new("RGBA", size)
    d = ImageDraw.Draw(im)
    for w, colour in ((width + 6, OUTLINE), (width, (*fill, 255))):
        d.line([a, b], fill=colour, width=w)
        for x, y in (a, b):
            d.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=colour)
    return np.array(im)


def over(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Porter-Duff 'src over dst' on float RGBA (0-255). / 標準的 alpha 疊圖。"""
    sa, da = src[..., 3:] / 255, dst[..., 3:] / 255
    a = sa + da * (1 - sa)
    rgb = (src[..., :3] * sa + dst[..., :3] * da * (1 - sa)) / np.maximum(a, 1e-6)
    return np.dstack([rgb, a * 255])


def turned(rgba: np.ndarray, pivot: tuple[float, float], degrees: float, shift=(0.0, 0.0)) -> np.ndarray:
    """Turn a layer about a pivot, then move it. Done on premultiplied colour so edges stay clean.
    圖層繞支點旋轉再平移。用預乘 alpha（RGBa）旋轉，邊緣才不會出現黑邊。"""
    im = Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), "RGBA").convert("RGBa")
    im = im.rotate(degrees, Image.BICUBIC, center=pivot, translate=shift)
    return np.array(im.convert("RGBA")).astype(float)


def recolour(rgba: np.ndarray, mask_skin: np.ndarray, target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Move skin from one lighting to another (lit near leg <-> shaded far leg), leaving shoes alone.
    把皮膚從一種光線換成另一種（亮的近側腿 <-> 暗的遠側腿），鞋子不動。"""
    out = rgba.copy()
    ratio = target / source
    out[..., :3] = np.where(mask_skin[..., None], np.clip(rgba[..., :3] * ratio, 0, 255), rgba[..., :3])
    return out


def skin_of(rgba: np.ndarray) -> np.ndarray:
    """Skin and sock pixels (anything clearly lighter than the brown shoes), softened at the edges.
    皮膚和襪子的像素（比咖啡色鞋子亮很多的部分），邊緣羽化。"""
    r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
    hard = (r > 170) & (g > 115) & (b > 100)
    return ndimage.gaussian_filter(hard.astype(float), 1.0) > 0.5


class Puppet:
    """The drawing cut into layers, ready to pose. / 拆成圖層、可以擺姿勢的紙娃娃。"""

    def __init__(self) -> None:
        src = np.array(Image.open(SOURCE).convert("RGBA")).astype(float)
        # Pad so swung limbs never fall off the canvas. / 加大畫布，手腳擺出去不會被裁掉。
        h, w = src.shape[:2]
        self.size = (w + 2 * PAD, h + 2 * PAD)
        full = np.zeros((h + 2 * PAD, w + 2 * PAD, 4))
        full[PAD:PAD + h, PAD:PAD + w] = src
        self.src = full
        p = lambda pts: [(x + PAD, y + PAD) for x, y in pts]          # noqa: E731 - source -> padded / 原圖座標 -> 加框座標
        self.p = lambda pt: (pt[0] + PAD, pt[1] + PAD)                  # noqa: E731
        r, g, b, a = (full[..., i] for i in range(4))

        # 1. The forearm: skin and brown outline inside ARM_BOX, the piece that touches the fist.
        #    前臂：ARM_BOX 範圍內的膚色和咖啡色線條，取跟拳頭相連的那一塊。
        box = polygon(p(ARM_BOX), self.size)
        skin = (r > 225) & (g > 165) & (b > 140) & (g < 242)
        line = (r < 215) & (r > g + 35) & (b < g + 10) & (g > 60)
        lab, _ = ndimage.label(box & (skin | line) & (a > 100))
        arm = lab == lab[650 + PAD, 1000 + PAD]
        arm = ndimage.binary_fill_holes(ndimage.binary_closing(arm, iterations=3))
        arm = ndimage.binary_dilation(arm, iterations=2) & (a > 0) & box
        forearm = layer(full, arm)
        # The upper arm is hidden in the sleeve; a skin capsule stands in for it so the arm stays joined.
        # 上臂藏在袖子裡；用一段膚色膠囊代替，手臂轉動時才會一直連著身體。
        upper = capsule(self.size, self.p(SHOULDER), self.p(ELBOW), 54, (250, 206, 188))
        self.arm = over(upper.astype(float), forearm)

        # 2. The body without the arm: paint what was behind it (inpaint), then keep only chest and hair.
        #    沒有手臂的身體：用 inpaint 補畫手臂後面的東西，再只保留胸口與頭髮的部分。
        hole = ndimage.binary_dilation(arm, iterations=3)
        rgb = cv2.inpaint(full[..., :3].astype(np.uint8), hole.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)
        behind = polygon(p(CHEST), self.size) | polygon(p(HAIR), self.size)
        alpha = np.where(hole, np.where(behind, 255.0, 0.0), a)
        alpha = np.where(hole, ndimage.gaussian_filter(alpha, 1.2), alpha)
        body = np.dstack([np.where(hole[..., None], rgb, full[..., :3]), alpha])

        # 3. Legs: inside each leg's box and under the hem. They come off the body.
        #    腿：在各自框內、且在裙擺下緣以下的像素。從身體上拿掉。
        hem = below(p(HEM), self.size)
        front = polygon(p(FRONT_LEG), self.size) & hem
        back = polygon(p(BACK_LEG), self.size) & hem
        body[..., 3] = np.where(front | back, 0, body[..., 3])
        self.body = body
        # Cut from the arm-less body, not the drawing, or a stub of the old forearm would ride along.
        # 從「已拿掉手臂」的身體剪，不是從原圖剪，不然會帶著一小截舊的前臂。
        self.sleeve = layer(body, polygon(p(SLEEVE), self.size))
        # Each leg gets a thigh capsule up to its hip, hidden under the skirt until the leg turns.
        # 每條腿補一段大腿到髖關節，平常藏在裙子下，腿轉動時才派上用場。
        self.front_leg = self._with_thigh(layer(full, front), FRONT_HIP, (930, 940), SKIN_FAR)
        self.back_leg = self._with_thigh(layer(full, back), BACK_HIP, (680, 925), SKIN_NEAR)
        # The same two drawings lit the other way round, for the half-stride with the near leg in front.
        # 同樣兩張腿換一種光影，給「近側腿在前」的那半步用。
        lit = recolour(self.front_leg, skin_of(self.front_leg), SKIN_NEAR, SKIN_FAR)
        sock = polygon(p(SOCK), self.size) & skin_of(self.front_leg)
        shade = lit[..., :3].mean(-1, keepdims=True) / SKIN_NEAR.mean()      # keep the drawing's shading / 保留原畫明暗
        soft = ndimage.gaussian_filter(sock.astype(float), 1.5)[..., None]
        lit[..., :3] = lit[..., :3] * (1 - soft) + np.clip(SOCK_WHITE * shade, 0, 255) * soft
        self.near_front_legs = (lit, recolour(self.back_leg, skin_of(self.back_leg), SKIN_FAR, SKIN_NEAR))

    def _with_thigh(self, leg: np.ndarray, hip, knee, skin) -> np.ndarray:
        thigh = capsule(self.size, self.p(hip), self.p(knee), 70, tuple(int(c) for c in skin)).astype(float)
        return over(thigh, leg)

    def legs(self, near_front: bool) -> tuple[np.ndarray, np.ndarray]:
        """The front and back leg, lit for whichever side is near. In the drawing the far leg is in front, so
        when the near leg comes forward, the two drawings just swap lighting and depth.
        前腿和後腿，依照哪一邊是近側來上光影。原畫是遠側腿在前；近側腿往前時，兩張腿只要互換明暗和前後順序。"""
        return self.near_front_legs if near_front else (self.front_leg, self.back_leg)

    def pose(self, f: dict, arm_phase: float) -> np.ndarray:
        """One frame. arm_phase 0 = near arm fully forward, 1 = fully back.
        畫出一格。arm_phase 0 = 近側手最前，1 = 最後。"""
        canvas = np.zeros((self.size[1], self.size[0], 4))
        down = (0, f["body"])
        near_arm = turned(self.arm, self.p(SHOULDER), ARM_SWING * arm_phase, down)
        # The far arm is the near arm in shadow, swinging the other way, behind the body.
        # 遠側手 = 近側手的暗色版，反方向擺，畫在身體後面。
        far = self.arm.copy()
        far[..., :3] *= np.array([0.86, 0.8, 0.8])
        far_arm = turned(far, self.p(SHOULDER), ARM_SWING * (1 - arm_phase),
                         (FAR_SHOULDER[0] - SHOULDER[0], FAR_SHOULDER[1] - SHOULDER[1] + f["body"]))
        front, back = self.legs(f["near_front"])
        front = turned(front, self.p(FRONT_HIP), f["front"])
        back = turned(back, self.p(BACK_HIP), f["back"])
        near_leg, far_leg = (front, back) if f["near_front"] else (back, front)
        # Back to front: far arm, far leg, near leg, body, near arm, sleeve.
        # 由後到前：遠側手、遠側腿、近側腿、身體、近側手、袖子。
        for part in (far_arm, far_leg, near_leg, turned(self.body, (0, 0), 0, down), near_arm,
                     turned(self.sleeve, (0, 0), 0, down)):
            canvas = over(canvas, part)
        return canvas


def cycle() -> list[Image.Image]:
    """The 8 frames, each standing on the same ground line, flight frames lifted.
    8 格畫面，都站在同一條地平線上，騰空的格子往上抬。"""
    puppet = Puppet()
    raw = []
    for i, f in enumerate(FRAMES):
        arm_phase = (1 - np.cos(2 * np.pi * i / len(FRAMES))) / 2
        raw.append(puppet.pose(f, arm_phase))
    ground = max(Image.fromarray(r[..., 3].astype(np.uint8)).getbbox()[3] for r in raw)
    frames = []
    for r, f in zip(raw, FRAMES, strict=True):
        im = Image.fromarray(np.clip(r, 0, 255).astype(np.uint8), "RGBA")
        feet = im.getchannel("A").getbbox()[3]
        frames.append(im.transform(im.size, Image.AFFINE, (1, 0, 0, 0, 1, feet - ground + f["lift"])))
    return frames


def preview(frames: list[Image.Image], path: Path = PREVIEW, height: int = 360) -> None:
    """A looping GIF on white, at about the size the game draws her. / 白底循環 GIF，大小接近遊戲中的尺寸。"""
    box = Image.new("L", frames[0].size)
    for f in frames:
        box.paste(255, mask=f.getchannel("A"))
    left, top, right, bottom = box.getbbox()
    k = height / (bottom - top)
    out = []
    for f in frames:
        bg = Image.new("RGBA", f.size, (255, 255, 255, 255))
        bg.alpha_composite(f)
        out.append(bg.crop((left, top, right, bottom)).resize((round((right - left) * k), height),
                                                              Image.LANCZOS).convert("RGB"))
    out[0].save(path, save_all=True, append_images=out[1:], duration=90, loop=0)


if __name__ == "__main__":
    preview(cycle())
    print(PREVIEW.relative_to(ROOT))
