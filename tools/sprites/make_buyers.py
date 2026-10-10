"""Generate the buyer lineup sprite sheet.
產生「買家角色一覽」的像素圖精靈表（sprite sheet）。

Every pixel is placed by code, so the art is reproducible, diffable and easy to tweak.
每個像素都由程式碼放置：結果可重現、可在 git 裡比對差異、也容易微調。

Output / 輸出:
    game/assets/sprites/buyers.png      512x64, 1 px = 1 game pixel   (遊戲實際載入用)
    game/assets/sprites/buyers@4x.png   2048x256, nearest-neighbour 4x (預覽 / 展示用)
    game/assets/sprites/buyers.json     cell rectangles per buyer      (每格角色的座標表)

Usage / 用法:
    python tools/sprites/make_buyers.py

Requires Pillow (`pip install pillow`).  需要 Pillow 套件。
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Sheet layout  表單版面
# ---------------------------------------------------------------------------
CELL = 64  # each buyer lives in its own 64x64 cell  每個角色獨佔一個 64x64 的格子
COLS = 8  # eight buyers in one row  一列八位買家
SCALE = 4  # preview zoom; integer so every pixel stays a crisp square  預覽倍率，整數倍才能保持方塊銳利
GROUND = 62  # last row the feet may touch; row 63 stays empty as the 1 px margin  腳底所在列，第 63 列留白當作 1px 邊界
MARGIN = 1  # empty pixels required on every side of a cell  每格四周至少要留的空白像素

OUT_DIR = Path(__file__).resolve().parents[2] / "game" / "assets" / "sprites"

# ---------------------------------------------------------------------------
# Shared palette (32 colours max).  共用色盤（最多 32 色）
# Colours come in base/shade pairs so cel shading never invents new colours.
# 顏色以「亮面 / 暗面」成對出現，賽璐璐上色時就不會冒出色盤以外的顏色。
# ---------------------------------------------------------------------------
PALETTE: dict[str, tuple[int, int, int]] = {
    "bg": (255, 0, 255),  # chroma-key magenta, treated as transparent  洋紅色去背色，視為透明
    "ink": (27, 20, 38),  # outline, pupils  外框線、瞳孔
    "white": (246, 242, 234),
    "skin": (248, 204, 164),
    "skin_s": (220, 150, 116),
    "tan": (184, 120, 80),  # second skin tone  第二種膚色
    "tan_s": (136, 84, 54),
    "pink": (244, 146, 168),
    "pink_s": (200, 92, 126),
    "red": (226, 60, 60),
    "red_s": (156, 34, 52),
    "mustard": (232, 176, 48),
    "mustard_s": (176, 120, 32),
    "orange": (244, 130, 44),
    "orange_s": (186, 80, 30),
    "brown": (150, 100, 62),
    "brown_s": (96, 62, 40),
    "silver": (200, 196, 210),
    "gray": (138, 134, 152),
    "slate": (72, 70, 90),
    "sky": (112, 186, 234),
    "sky_s": (62, 124, 186),
    "navy": (52, 64, 122),
    "navy_s": (32, 40, 82),
    "green": (76, 160, 80),
    "green_s": (44, 104, 60),
    "plum": (148, 82, 178),
    "plum_s": (96, 48, 128),
    "yellow": (250, 222, 72),
    "khaki": (218, 194, 138),
    "khaki_s": (168, 140, 92),
    "hair_k": (52, 44, 66),  # "black" hair, lighter than ink so the outline still reads  黑髮，比外框淺一階才看得出輪廓
}
assert len(PALETTE) <= 32, "palette must stay within 32 colours"


# ---------------------------------------------------------------------------
# Mask helpers: shapes are sets of (x, y) pixels.
# 形狀工具：每個形狀都是一組 (x, y) 像素集合。
# Pillow draws into a 1-bit image, which never anti-aliases.
# 用 Pillow 畫在 1-bit 影像上，天生不會有反鋸齒（邊緣永遠是硬的）。
# ---------------------------------------------------------------------------
Mask = set[tuple[int, int]]


def _mask(draw_fn) -> Mask:
    img = Image.new("1", (CELL, CELL), 0)
    draw_fn(ImageDraw.Draw(img))
    pix = img.load()
    return {(x, y) for y in range(CELL) for x in range(CELL) if pix[x, y]}


def ellipse(x0: int, y0: int, x1: int, y1: int) -> Mask:
    return _mask(lambda d: d.ellipse((x0, y0, x1, y1), fill=1))


def rect(x0: int, y0: int, x1: int, y1: int) -> Mask:
    return _mask(lambda d: d.rectangle((x0, y0, x1, y1), fill=1))


def poly(*pts: tuple[int, int]) -> Mask:
    return _mask(lambda d: d.polygon(list(pts), fill=1))


def line(*pts: tuple[int, int]) -> Mask:
    return _mask(lambda d: d.line(list(pts), fill=1, width=1))


# ---------------------------------------------------------------------------
# Sprite: one 64x64 cell, built up in layers from back to front.
# Sprite：一個 64x64 格子，由後往前一層一層疊上去（畫家演算法）。
# ---------------------------------------------------------------------------
class Sprite:
    def __init__(self) -> None:
        self.px: dict[tuple[int, int], str] = {}  # pixel -> palette key  像素 -> 色盤名稱

    def fill(
        self,
        m: Mask,
        color: str,
        shade: str | None = None,
        sw: int = 2,
        sb: int = 1,
        hl: str | None = None,
        edge: str | None = "ink",
    ) -> None:
        """Paint a part on top of what is already there.
        把一個部件畫在現有圖層之上。

        shade/sw/sb: light comes from the upper left, so the right `sw` and bottom `sb`
                     pixels of the part get the shade colour.
                     光源在左上，部件右側 sw 格、下緣 sb 格塗暗色。
        hl:          1 px highlight along the top edge.  沿上緣 1px 亮光。
        edge:        colour for the inner line where this part overlaps an earlier one,
                     which gives the sticker-like line work of the reference style.
                     與先前部件重疊處的內線顏色，做出參考圖那種貼紙般的線條感。
        """
        under = set(self.px)
        for x, y in m:
            c = color
            if hl and (x, y - 1) not in m:
                c = hl
            if shade and (
                any((x + i, y) not in m for i in range(1, sw + 1))
                or any((x, y + j) not in m for j in range(1, sb + 1))
            ):
                c = shade
            if edge and any(
                n in under and n not in m for n in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
            ):
                c = edge
            self.px[(x, y)] = c

    def dot(self, color: str, *pts: tuple[int, int]) -> None:
        """Set single pixels directly (eyes, buttons, glints).  直接點像素（眼睛、鈕扣、反光）。"""
        for p in pts:
            self.px[p] = color

    def paint(self, m: Mask, color: str) -> None:
        """Flat fill with no shading or lines.  純色平塗，不加陰影或內線。"""
        for p in m:
            self.px[p] = color

    def outline(self) -> None:
        """Add the outer 1 px ink outline around the silhouette.
        在剪影外圍加上 1px 外框線。

        4-neighbour (not 8) keeps the line exactly one pixel thick with no corner blobs.
        只看上下左右 4 鄰居（不看斜角），線條才會剛好 1px 粗、轉角不會結塊。
        """
        ring = set()
        for x, y in self.px:
            for n in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if n not in self.px:
                    ring.add(n)
        for p in ring:
            self.px[p] = "ink"

    def check(self, name: str) -> None:
        """Enforce the cell rules so a bad edit fails loudly instead of shipping.
        檢查格子規則：違規就直接報錯，而不是默默輸出壞圖。"""
        xs = [x for x, _ in self.px]
        ys = [y for _, y in self.px]
        lo, hi = MARGIN, CELL - 1 - MARGIN
        assert min(xs) >= lo and max(xs) <= hi, f"{name}: x range {min(xs)}..{max(xs)} leaves cell"
        assert min(ys) >= lo, f"{name}: top at y={min(ys)} touches the cell edge"
        assert max(ys) == GROUND, f"{name}: feet at y={max(ys)}, expected ground y={GROUND}"


# ---------------------------------------------------------------------------
# Shared body parts.  共用的身體部件
# All characters face right in three-quarter view: the face sits on the right half
# of the head, the far eye is closer to the head's right edge, the ear is on the left.
# 所有角色都朝右、四分之三側面：五官偏向頭的右半邊，遠側眼睛靠近右緣，耳朵在左側。
# ---------------------------------------------------------------------------
def head(s: Sprite, x0: int, y0: int, w: int = 24, h: int = 22, skin: str = "skin") -> None:
    s.fill(ellipse(x0, y0, x0 + w, y0 + h), skin, f"{skin}_s", sw=1, sb=2, edge=None)


def ear(s: Sprite, x: int, y: int, skin: str = "skin") -> None:
    s.fill(rect(x, y, x + 2, y + 3), skin, f"{skin}_s", sw=1, sb=0)
    s.dot(f"{skin}_s", (x + 1, y + 1), (x + 1, y + 2))


def eyes(s: Sprite, x: int, y: int, style: str = "open") -> None:
    """Near eye at x, far eye at x+6.  近側眼在 x，遠側眼在 x+6。

    Chibi eyes are tall dark blocks with a white glint; that glint is what makes a
    64 px face read as "looking at you".
    Q 版眼睛是直長的深色塊加一個白色反光點；64px 的臉就靠這個反光點顯得有神。
    """
    if style == "open":
        for ex, h in ((x, 4), (x + 6, 4)):
            s.paint(rect(ex, y, ex + 1, y + h - 1), "ink")
            s.dot("white", (ex, y))
    elif style == "tired":  # half-lidded, with eye bags  半閉的眼皮加黑眼圈
        for ex in (x, x + 6):
            s.paint(rect(ex - 1, y + 1, ex + 2, y + 1), "ink")
            s.paint(rect(ex, y + 2, ex + 1, y + 2), "ink")
            s.dot("plum_s", (ex, y + 3), (ex + 1, y + 3))
    elif style == "happy":  # ^ ^ closed arcs  瞇眼笑 ^ ^
        for ex in (x, x + 6):
            s.dot("ink", (ex - 1, y + 2), (ex, y + 1), (ex + 1, y + 1), (ex + 2, y + 2))
    elif style == "determined":  # open eyes under slanted brows  斜眉 + 睜眼，表示「志在必得」
        for ex in (x, x + 6):
            s.paint(rect(ex, y + 1, ex + 1, y + 3), "ink")
            s.dot("white", (ex + 1, y + 1))
        s.dot("ink", (x - 1, y - 2), (x, y - 2), (x + 1, y - 1), (x + 2, y - 1))
        s.dot("ink", (x + 5, y - 1), (x + 6, y - 1), (x + 7, y - 2), (x + 8, y - 2))


def blush(s: Sprite, x: int, y: int) -> None:
    s.dot("pink", (x - 2, y), (x - 1, y), (x + 8, y))


# ---------------------------------------------------------------------------
# The eight buyers.  八位買家
# Each function draws back-to-front: back arm / bag -> legs -> body -> front arm -> head
# -> hair -> props.  每個函式都由後往前畫：後手或背包 -> 腿 -> 身體 -> 前手 -> 頭 -> 頭髮 -> 道具。
# ---------------------------------------------------------------------------
def grandpa() -> Sprite:
    """Retired grandfather: mustard cardigan, flat cap, cane.  退休爺爺：芥末黃開襟衫、鴨舌帽、拐杖。"""
    s = Sprite()
    # cane, behind the hand  拐杖（在手後面）
    s.fill(rect(44, 38, 45, 61), "brown_s", edge=None)
    s.fill(rect(41, 37, 45, 38), "brown_s", edge=None)
    # legs: gray slacks, brown shoes  腿：灰色長褲、棕色皮鞋
    s.fill(rect(25, 47, 30, 58), "gray", "slate", sw=1, sb=0, edge=None)
    s.fill(rect(32, 47, 37, 58), "gray", "slate", sw=1, sb=0)
    s.fill(rect(24, 58, 32, 61), "brown", "brown_s", sw=1, sb=1, edge=None)
    s.fill(rect(32, 58, 40, 61), "brown", "brown_s", sw=1, sb=1)
    # back arm  後手
    s.fill(rect(37, 29, 41, 40), "mustard_s", edge=None)
    # cardigan body, slightly rounded (a little hunched)  開襟衫身體，略圓（有點駝背）
    s.fill(poly((24, 27), (38, 27), (40, 48), (22, 48)), "mustard", "mustard_s", sw=2, sb=1)
    s.fill(poly((31, 27), (35, 27), (33, 36)), "white", edge="ink")  # shirt collar 襯衫領
    s.dot("brown_s", (36, 34), (36, 39), (36, 44))  # buttons 鈕扣
    s.fill(rect(25, 41, 30, 44), "mustard_s", edge=None)  # pocket 口袋
    # front arm reaching forward to the cane  前手伸向拐杖
    s.fill(poly((24, 29), (29, 29), (42, 35), (42, 39), (28, 37), (24, 35)), "mustard", "mustard_s", sw=1, sb=1)
    s.fill(rect(41, 35, 44, 39), "skin", "skin_s", sw=1, sb=1)  # hand 手
    # head  頭
    head(s, 19, 6)
    ear(s, 23, 15)
    # silver hair at the back and side  後側銀白頭髮
    s.fill(poly((19, 12), (25, 10), (25, 20), (21, 24), (19, 20)), "silver", "gray", sw=1, sb=1)
    # flat cap with a brim pointing forward  帽簷朝前的鴨舌帽
    s.fill(poly((18, 13), (19, 7), (25, 3), (36, 3), (43, 8), (44, 12)), "brown", "brown_s", sw=0, sb=2, hl=None)
    s.fill(poly((38, 10), (47, 11), (47, 13), (38, 13)), "brown_s")
    # round glasses, face  圓框眼鏡、五官
    eyes(s, 33, 15, "happy")
    s.paint(rect(31, 14, 35, 18) - rect(32, 15, 34, 17), "slate")
    s.paint(rect(37, 14, 41, 18) - rect(38, 15, 40, 17), "slate")
    s.dot("slate", (36, 15))
    s.paint(rect(34, 21, 41, 22), "silver")  # mustache 鬍子
    s.dot("gray", (35, 22), (40, 22))
    s.dot("skin_s", (42, 18))  # nose tip on the profile edge  側面的鼻尖
    s.outline()
    return s


def student() -> Sprite:
    """College student: red hoodie, big navy backpack.  大學生：紅色帽T、深藍大背包。"""
    s = Sprite()
    # backpack on the back (left side, since they face right)  背包在背後（朝右站，所以在左邊）
    s.fill(poly((13, 28), (22, 25), (25, 26), (25, 47), (14, 48), (12, 44)), "navy", "navy_s", sw=1, sb=2)
    s.fill(rect(13, 37, 18, 43), "navy_s", edge=None)  # front pocket  前袋
    s.dot("yellow", (15, 38), (16, 38))  # zipper pull  拉鍊頭
    # jeans + sneakers  牛仔褲 + 球鞋
    s.fill(rect(25, 46, 30, 57), "sky_s", "navy", sw=1, sb=0, edge=None)
    s.fill(rect(32, 46, 37, 57), "sky_s", "navy", sw=1, sb=0)
    s.fill(rect(24, 57, 32, 61), "white", "silver", sw=0, sb=1, edge=None)
    s.fill(rect(33, 57, 41, 61), "white", "silver", sw=0, sb=1)
    s.paint(rect(24, 61, 32, 61) | rect(33, 61, 41, 61), "red_s")  # soles 鞋底
    # back arm  後手
    s.fill(rect(37, 29, 41, 41), "red_s", edge=None)
    s.fill(rect(38, 41, 41, 43), "skin_s", edge=None)
    # hoodie body with the hood bunched behind the neck  帽T身體，帽子堆在後頸
    s.fill(poly((24, 27), (38, 27), (39, 47), (23, 47)), "red", "red_s", sw=2, sb=1)
    s.fill(poly((21, 24), (30, 24), (31, 30), (22, 31)), "red", "red_s", sw=1, sb=1)  # hood 帽子
    s.fill(rect(26, 40, 36, 44), "red_s")  # kangaroo pocket 袋鼠口袋
    s.dot("white", (34, 29), (34, 30), (34, 31), (36, 29), (36, 30))  # drawstrings 帽繩
    # backpack strap over the front shoulder  背包肩帶繞過前肩
    s.fill(poly((24, 27), (27, 27), (28, 42), (25, 42)), "navy", "navy_s", sw=1, sb=0)
    # front arm, hand in pocket  前手插口袋
    s.fill(poly((25, 29), (30, 29), (32, 40), (27, 41)), "red", "red_s", sw=1, sb=0)
    # head  頭
    head(s, 19, 5)
    ear(s, 24, 15)
    # messy black hair with spikes  亂翹的黑髮
    s.fill(
        poly(
            (18, 18), (17, 10), (20, 5), (24, 2), (27, 4), (30, 2), (33, 4), (37, 3),
            (39, 6), (44, 7), (43, 13), (40, 11), (38, 14), (35, 11), (31, 14), (28, 12), (26, 20), (22, 22),
        ),
        "hair_k", "ink", sw=0, sb=1, hl="slate",
    )
    eyes(s, 33, 15, "open")
    blush(s, 33, 20)
    s.paint(rect(37, 22, 38, 22), "ink")  # small smile 小微笑
    s.dot("skin_s", (43, 18))
    s.outline()
    return s


def nurse() -> Sprite:
    """Nurse after a night shift: sky-blue scrubs, tired eyes, coffee.  下大夜班的護理師：天藍色制服、疲倦的眼、咖啡。"""
    s = Sprite()
    # scrub pants + white clogs  制服褲 + 白色護士鞋
    s.fill(rect(25, 46, 30, 58), "sky", "sky_s", sw=1, sb=0, edge=None)
    s.fill(rect(32, 46, 37, 58), "sky", "sky_s", sw=1, sb=0)
    s.fill(rect(24, 58, 31, 61), "white", "silver", sw=0, sb=1, edge=None)
    s.fill(rect(32, 58, 39, 61), "white", "silver", sw=0, sb=1)
    # back arm  後手
    s.fill(rect(36, 29, 40, 38), "sky_s", edge=None)
    s.fill(rect(37, 38, 40, 41), "skin_s", edge=None)
    # scrub top, V-neck  V 領制服上衣
    s.fill(poly((24, 27), (38, 27), (39, 47), (23, 47)), "sky", "sky_s", sw=2, sb=1)
    s.fill(poly((30, 27), (35, 27), (32, 32)), "skin", edge="ink")
    s.fill(rect(26, 38, 30, 42), "sky_s")  # chest pocket 口袋
    s.dot("red", (27, 37), (28, 37))  # pen in the pocket 口袋裡的筆
    # stethoscope around the neck  脖子上的聽診器
    s.paint(line((27, 27), (28, 33), (31, 34)) | line((36, 27), (35, 32)), "slate")
    s.dot("silver", (31, 35), (32, 35), (35, 33))
    # ID badge on a lanyard  識別證
    s.fill(rect(33, 36, 36, 40), "white", edge="ink")
    s.dot("sky_s", (34, 37), (35, 37))
    # front arm raised, holding a takeaway coffee  前手舉著外帶咖啡
    s.fill(poly((25, 29), (30, 29), (38, 36), (35, 39), (26, 36)), "sky", "sky_s", sw=1, sb=1)
    s.fill(rect(37, 33, 40, 37), "skin", "skin_s", sw=1, sb=1)
    s.fill(poly((39, 27), (45, 27), (44, 37), (40, 37)), "white", "silver", sw=1, sb=0)  # cup 杯子
    s.fill(rect(39, 31, 45, 33), "brown", edge=None)  # sleeve 杯套
    s.fill(rect(38, 25, 46, 27), "slate", edge=None)  # lid 杯蓋
    s.dot("silver", (41, 23), (42, 22), (42, 21))  # steam 熱氣
    # head  頭
    head(s, 19, 5)
    ear(s, 23, 15)
    # brown hair in a slightly messy bun  有點亂的棕色包包頭
    s.fill(ellipse(15, 3, 23, 11), "brown", "brown_s", sw=1, sb=1)  # bun 髮髻
    s.fill(
        poly((18, 21), (17, 11), (21, 5), (28, 3), (36, 4), (42, 9), (44, 14), (39, 11), (34, 13), (30, 10), (27, 14), (25, 23)),
        "brown", "brown_s", sw=0, sb=1,
    )
    s.dot("brown", (25, 2), (24, 2), (44, 15), (45, 16))  # loose strands 散落的髮絲
    eyes(s, 33, 15, "tired")
    s.paint(rect(36, 22, 38, 23), "ink")  # small yawn 小呵欠
    s.dot("skin_s", (43, 18))
    s.outline()
    return s


def mother_and_child() -> Sprite:
    """Mother in a pink cardigan holding her child's hand.  穿粉色開襟衫的媽媽牽著小孩。

    Both share one cell, so the mother stands left and the child sits in the free space
    on the right; their hands meet in the middle.
    兩人共用一格：媽媽站左邊、小孩在右邊空位，兩隻手在中間牽在一起。
    """
    s = Sprite()
    # --- mother (left)  媽媽（左） ---
    # long ponytail falling behind  垂在背後的長馬尾
    s.fill(poly((8, 14), (13, 12), (14, 32), (11, 38), (8, 34)), "brown", "brown_s", sw=1, sb=1)
    # navy trousers + flats  深藍長褲 + 平底鞋
    s.fill(rect(17, 46, 22, 58), "navy", "navy_s", sw=1, sb=0, edge=None)
    s.fill(rect(24, 46, 29, 58), "navy", "navy_s", sw=1, sb=0)
    s.fill(rect(16, 58, 23, 61), "brown_s", edge=None)
    s.fill(rect(24, 58, 31, 61), "brown_s")
    # cardigan body over a white top  白上衣外罩粉色開襟衫
    s.fill(poly((16, 27), (30, 27), (32, 48), (14, 48)), "pink", "pink_s", sw=2, sb=1)
    s.fill(poly((24, 27), (29, 27), (28, 47), (26, 47)), "white", "silver", sw=1, sb=0)
    # shoulder bag strap  斜背包帶
    s.paint(line((17, 27), (29, 43)), "brown_s")
    # front arm reaching down to the child's hand  前手往下牽住小孩
    s.fill(poly((17, 29), (22, 29), (36, 40), (34, 43), (18, 37)), "pink", "pink_s", sw=1, sb=1)
    s.fill(rect(34, 40, 37, 43), "skin", "skin_s", sw=1, sb=1)
    # head  頭
    head(s, 11, 5)
    ear(s, 15, 15)
    s.fill(
        poly((10, 21), (9, 11), (13, 5), (21, 3), (29, 4), (34, 8), (36, 14), (31, 11), (27, 13), (24, 10), (21, 13), (18, 22)),
        "brown", "brown_s", sw=0, sb=1,
    )
    s.fill(ellipse(9, 8, 13, 12), "pink", edge="ink")  # scrunchie 髮圈
    eyes(s, 25, 15, "happy")
    blush(s, 25, 19)
    s.dot("ink", (29, 21), (30, 22), (31, 22), (32, 21))  # smile 微笑
    s.dot("skin_s", (35, 18))
    # --- child (right)  小孩（右） ---
    # red rain boots + legs  紅色雨鞋
    s.fill(rect(42, 54, 45, 61), "red", "red_s", sw=1, sb=1, edge=None)
    s.fill(rect(47, 54, 50, 61), "red", "red_s", sw=1, sb=1)
    # yellow raincoat, a small A-line  黃色小雨衣（A 字形）
    s.fill(poly((42, 41), (51, 41), (54, 55), (40, 55)), "yellow", "mustard", sw=2, sb=1)
    s.dot("mustard_s", (48, 45), (48, 49))  # toggles 牛角扣
    # child's raised hand meeting mom's hand  小孩舉起的手跟媽媽的手相握
    s.fill(poly((41, 42), (44, 42), (40, 46), (37, 44)), "yellow", "mustard", sw=1, sb=1)
    s.fill(rect(36, 41, 38, 43), "skin", "skin_s", sw=1, sb=0, edge=None)
    # child's head, hood up  小孩的頭（戴著雨帽）
    s.fill(ellipse(39, 25, 57, 42), "yellow", "mustard", sw=1, sb=1)  # hood 雨帽
    s.fill(ellipse(43, 28, 57, 42), "skin", "skin_s", sw=1, sb=2)
    s.fill(poly((43, 28), (55, 28), (57, 32), (47, 31), (43, 34)), "brown", "brown_s", sw=0, sb=1)  # bangs 瀏海
    s.paint(rect(49, 34, 49, 36) | rect(54, 34, 54, 36), "ink")
    s.dot("white", (49, 34), (54, 34))
    s.dot("pink", (47, 38), (48, 38), (56, 38))
    s.dot("ink", (51, 39), (52, 39))
    s.outline()
    return s


def fashionista() -> Sprite:
    """Fashionable young woman: plum long coat, sunglasses, long dark hair.
    時髦年輕女性：梅紫色長大衣、墨鏡、黑長髮。"""
    s = Sprite()
    # long hair behind the shoulders  披在肩後的長髮
    s.fill(poly((17, 12), (24, 10), (26, 38), (21, 40), (16, 36), (18, 26)), "hair_k", "ink", sw=1, sb=1, hl=None)
    # boots with a small heel  有小跟的靴子
    s.fill(rect(26, 50, 29, 58), "slate", edge=None)
    s.fill(rect(32, 50, 35, 58), "slate")
    s.fill(poly((25, 58), (31, 58), (32, 61), (25, 61)), "ink", edge=None)
    s.fill(poly((31, 58), (37, 58), (38, 61), (31, 61)), "ink")
    s.dot("slate", (26, 61), (32, 61))  # heels 鞋跟
    # flared long coat to the knee  到膝蓋、下擺外擴的長大衣
    s.fill(poly((23, 27), (37, 27), (41, 51), (19, 51)), "plum", "plum_s", sw=2, sb=1)
    s.fill(poly((31, 27), (36, 27), (32, 37)), "white", edge="ink")  # blouse 襯衫
    s.paint(line((31, 27), (34, 40)), "plum_s")  # lapel 翻領
    s.fill(rect(20, 39, 40, 40), "plum_s", edge=None)  # belt 腰帶
    s.dot("mustard", (34, 39), (34, 40))  # buckle 釦環
    # front arm: hand on hip, holding a little handbag  前手叉腰、勾著小手提包
    s.fill(poly((23, 29), (28, 29), (27, 38), (21, 42), (18, 39)), "plum", "plum_s", sw=1, sb=1)
    s.fill(rect(18, 39, 21, 42), "skin", "skin_s", sw=1, sb=1, edge=None)
    s.fill(poly((13, 43), (21, 43), (22, 49), (12, 49)), "mustard", "mustard_s", sw=1, sb=1)  # bag 包包
    s.paint(line((15, 43), (17, 41), (19, 43)), "mustard_s")  # bag handle 提把
    # back arm  後手
    s.fill(rect(37, 29, 40, 41), "plum", "plum_s", sw=1, sb=0, edge=None)
    s.fill(rect(38, 41, 40, 43), "skin_s", edge=None)
    # head  頭
    head(s, 19, 5)
    ear(s, 24, 15)
    # sleek black hair with a side part and a wave over the face  側分黑長髮，前髮微卷
    s.fill(
        poly((18, 22), (17, 11), (21, 5), (29, 3), (37, 5), (42, 9), (44, 15), (40, 12), (33, 9), (28, 12), (26, 22)),
        "hair_k", "ink", sw=0, sb=1, hl="slate",
    )
    # oversized sunglasses  超大墨鏡
    s.paint(rect(31, 14, 36, 18) | rect(38, 14, 42, 18), "ink")
    s.dot("ink", (37, 15))
    s.dot("white", (32, 15), (33, 15), (39, 15))
    s.paint(rect(36, 22, 39, 22), "red")  # red lips 紅唇
    s.dot("skin_s", (43, 19))
    s.outline()
    return s


def hiker() -> Sprite:
    """Hiker: green shell jacket, cap, chunky boots, trekking pole.
    登山客：綠色衝鋒衣、棒球帽、厚登山靴、登山杖。"""
    s = Sprite()
    # trekking pole, behind the hand  登山杖
    s.fill(line((45, 34), (47, 61)), "slate", edge=None)
    s.fill(line((44, 34), (46, 61)), "silver", edge=None)
    # big backpack with a rolled sleeping mat on top  大背包 + 上方捲起的睡墊
    s.fill(poly((13, 27), (24, 25), (24, 48), (14, 49)), "orange_s", "brown_s", sw=1, sb=1)
    s.fill(rect(11, 21, 25, 26), "orange", "orange_s", sw=0, sb=1)  # sleeping mat 睡墊
    s.dot("orange_s", (14, 22), (18, 22), (22, 22))
    # khaki pants + chunky boots  卡其長褲 + 厚靴
    s.fill(rect(25, 46, 30, 55), "khaki", "khaki_s", sw=1, sb=0, edge=None)
    s.fill(rect(32, 46, 37, 55), "khaki", "khaki_s", sw=1, sb=0)
    s.fill(rect(24, 55, 32, 61), "brown", "brown_s", sw=1, sb=1, edge=None)
    s.fill(rect(32, 55, 41, 61), "brown", "brown_s", sw=1, sb=1)
    s.paint(rect(24, 61, 32, 61) | rect(32, 61, 41, 61), "ink")  # lug soles 鞋底紋
    s.dot("mustard", (27, 56), (35, 56))  # laces 鞋帶
    # back arm  後手
    s.fill(rect(37, 29, 41, 40), "green_s", edge=None)
    # jacket  外套
    s.fill(poly((24, 27), (38, 27), (39, 47), (23, 47)), "green", "green_s", sw=2, sb=1)
    s.paint(line((34, 27), (34, 46)), "green_s")  # zipper 拉鍊
    s.fill(rect(23, 44, 39, 47), "green_s", edge=None)  # hem 下擺
    s.paint(line((25, 27), (26, 43)), "orange_s")  # pack strap 背包肩帶
    # front arm forward, gripping the pole  前手往前握住登山杖
    s.fill(poly((25, 29), (30, 29), (43, 33), (43, 37), (28, 37)), "green", "green_s", sw=1, sb=1)
    s.fill(rect(42, 32, 46, 36), "tan", "tan_s", sw=1, sb=1)
    # head  頭
    head(s, 19, 5, skin="tan")
    s.fill(poly((18, 18), (18, 11), (26, 10), (26, 15), (22, 19)), "hair_k", "slate", sw=0, sb=1)  # short hair 短髮
    # cap with a forward brim  帽簷朝前的棒球帽
    s.fill(ellipse(18, 2, 42, 18) & rect(0, 0, 63, 11), "mustard", "mustard_s", sw=0, sb=1)
    s.fill(poly((36, 9), (48, 10), (48, 12), (36, 12)), "mustard_s")
    s.dot("mustard_s", (30, 3))  # top button 帽頂鈕
    ear(s, 24, 14, skin="tan")
    eyes(s, 33, 15, "open")
    s.paint(line((35, 21), (36, 22), (39, 22), (40, 21)), "ink")  # grin 笑
    s.dot("tan_s", (43, 18))
    s.outline()
    return s


def breeder() -> Sprite:
    """Cat-show breeder: neat navy blazer, rosette pin, holding a white show cat.
    貓展繁育者：整齊的深藍西裝外套、別著獎章花結、抱著白色參展貓。"""
    s = Sprite()
    # trousers + shoes  西裝褲 + 皮鞋
    s.fill(rect(25, 46, 30, 58), "navy_s", "ink", sw=1, sb=0, edge=None)
    s.fill(rect(32, 46, 37, 58), "navy_s", "ink", sw=1, sb=0)
    s.fill(rect(24, 58, 32, 61), "ink", edge=None)
    s.fill(rect(32, 58, 40, 61), "ink")
    s.dot("slate", (37, 58), (29, 58))  # shoe shine 皮鞋反光
    # back arm  後手
    s.fill(rect(37, 29, 41, 40), "navy_s", edge=None)
    # tailored blazer  合身西裝外套
    s.fill(poly((24, 27), (38, 27), (39, 48), (23, 48)), "navy", "navy_s", sw=2, sb=1)
    s.fill(poly((30, 27), (36, 27), (33, 38)), "white", edge="ink")  # shirt 襯衫
    s.paint(line((29, 27), (33, 38)) | line((37, 27), (34, 37)), "navy_s")  # lapels 翻領
    # white show cat held in the arms  抱在懷裡的白色參展貓
    s.fill(ellipse(29, 36, 45, 46), "white", "silver", sw=2, sb=2)  # body 身體
    s.fill(ellipse(38, 31, 48, 40), "white", "silver", sw=1, sb=1)  # head 頭
    s.fill(poly((39, 33), (39, 28), (42, 32)), "white", "silver", sw=0, sb=0)  # ear 耳朵
    s.fill(poly((44, 32), (47, 28), (47, 33)), "white", "silver", sw=0, sb=0)
    s.dot("pink", (40, 31), (46, 31))
    s.dot("ink", (42, 35), (46, 35), (44, 37))
    s.paint(line((30, 44), (27, 46), (26, 49)), "silver")  # tail 尾巴
    # front arm wrapped under the cat  前手托著貓
    s.fill(poly((24, 30), (29, 30), (30, 42), (40, 43), (40, 46), (27, 46), (24, 42)), "navy", "navy_s", sw=1, sb=1)
    s.fill(rect(39, 42, 42, 45), "skin", "skin_s", sw=1, sb=1)
    # rosette pinned over the lapel, drawn last so the arm never hides it
    # 獎章花結別在翻領上，最後才畫，手臂就不會把它蓋住
    s.paint(rect(25, 34, 25, 38) | rect(28, 34, 28, 37), "red_s")  # ribbon tails 緞帶
    s.fill(ellipse(23, 28, 29, 34), "yellow", "mustard", sw=1, sb=1)
    s.dot("red", (25, 30), (26, 30), (25, 31), (26, 31), (25, 32), (26, 32))
    # head  頭
    head(s, 19, 5)
    ear(s, 23, 15)
    # neat bob haircut  整齊的鮑伯頭
    s.fill(
        poly((17, 23), (17, 11), (22, 5), (30, 3), (38, 5), (43, 10), (44, 14), (34, 12), (28, 12), (27, 24)),
        "brown_s", "ink", sw=0, sb=1, hl="brown",
    )
    eyes(s, 33, 15, "open")
    blush(s, 33, 20)
    s.paint(rect(37, 22, 38, 22), "ink")
    s.dot("skin_s", (43, 18))
    s.outline()
    return s


def bargain_hunter() -> Sprite:
    """Thrifty bargain hunter: orange puffer vest, tote bag with a leek, coupon held up.
    精打細算的撿便宜達人：橘色鋪棉背心、插著蔥的帆布袋、高舉折價券。"""
    s = Sprite()
    # big canvas tote hanging behind, a leek sticking out  背後的大帆布袋，插著一根蔥
    s.fill(poly((15, 20), (17, 19), (21, 33), (19, 34)), "green", "green_s", sw=1, sb=0)  # leek 蔥
    s.fill(rect(15, 17, 17, 20), "white", edge=None)
    s.fill(poly((11, 32), (24, 32), (25, 48), (10, 48)), "khaki", "khaki_s", sw=2, sb=1)
    s.paint(rect(14, 38, 20, 41), "red")  # print on the tote 帆布袋上的印花
    s.dot("white", (16, 39), (18, 40))
    # brown slacks + sneakers  棕色長褲 + 球鞋
    s.fill(rect(25, 46, 30, 57), "brown", "brown_s", sw=1, sb=0, edge=None)
    s.fill(rect(32, 46, 37, 57), "brown", "brown_s", sw=1, sb=0)
    s.fill(rect(24, 57, 32, 61), "white", "silver", sw=0, sb=1, edge=None)
    s.fill(rect(33, 57, 41, 61), "white", "silver", sw=0, sb=1)
    # back arm  後手
    s.fill(rect(37, 29, 41, 40), "mustard_s", edge=None)
    # mustard sweater under an orange puffer vest  芥末黃毛衣 + 橘色鋪棉背心
    s.fill(poly((24, 27), (38, 27), (39, 47), (23, 47)), "orange", "orange_s", sw=2, sb=1)
    s.paint(rect(24, 33, 38, 33) | rect(24, 39, 39, 39), "orange_s")  # puffer quilting 鋪棉縫線
    s.fill(poly((31, 27), (35, 27), (34, 46), (32, 46)), "mustard", "mustard_s", sw=1, sb=0)
    # tote strap over the shoulder  帆布袋背帶
    s.fill(poly((22, 27), (25, 27), (24, 33), (21, 33)), "khaki", "khaki_s", sw=1, sb=0)
    # front arm raised high, waving a coupon  前手高舉折價券
    s.fill(poly((25, 29), (30, 29), (44, 23), (47, 26), (31, 36), (26, 36)), "mustard", "mustard_s", sw=1, sb=1)
    s.fill(rect(45, 20, 49, 25), "skin", "skin_s", sw=1, sb=1)
    # coupon with a perforated (every other pixel) red border  折價券，紅色虛線框（隔一格點一個）
    coupon = rect(47, 9, 58, 18)
    s.fill(coupon, "white", edge=None)
    s.paint((coupon - rect(48, 10, 57, 17)) & {(x, y) for x in range(CELL) for y in range(CELL) if (x + y) % 2 == 0}, "red")
    s.paint(rect(50, 12, 55, 12) | rect(50, 14, 53, 14), "red_s")  # price text 價格字樣
    s.fill(rect(46, 19, 49, 21), "skin", "skin_s", sw=1, sb=0, edge=None)  # thumb on top 拇指壓在券上
    # head  頭
    head(s, 18, 5)
    ear(s, 22, 15)
    # permed curly hair  燙捲的頭髮
    # A solid cap first so no skin peeks between the curls, then round bumps on top.
    # 先鋪一整片髮帽避免捲髮之間露出頭皮，再疊上一顆顆圓形捲髮。
    curls = (ellipse(17, 4, 43, 18) & rect(0, 0, 63, 10)) | poly((17, 9), (24, 9), (24, 22), (17, 22))
    for cx, cy in ((19, 9), (24, 6), (30, 5), (36, 6), (41, 8), (18, 14), (19, 19), (22, 21)):
        curls |= ellipse(cx - 3, cy - 3, cx + 3, cy + 3)
    for cx in (27, 31, 35, 39):  # curly fringe 捲捲的瀏海
        curls |= ellipse(cx - 2, 9, cx + 2, 12)
    s.fill(curls, "plum", "plum_s", sw=1, sb=1)
    eyes(s, 32, 15, "determined")
    s.paint(rect(35, 21, 39, 22), "ink")  # big grin 大笑
    s.paint(rect(36, 21, 38, 21), "white")  # teeth 牙齒
    s.dot("skin_s", (42, 18))
    s.dot("white", (45, 7), (46, 6), (60, 6), (60, 4))  # "deal!" sparkles 撿到便宜的閃光
    s.outline()
    return s


BUYERS = [
    ("grandpa", grandpa),
    ("student", student),
    ("nurse", nurse),
    ("mother_and_child", mother_and_child),
    ("fashionista", fashionista),
    ("hiker", hiker),
    ("breeder", breeder),
    ("bargain_hunter", bargain_hunter),
]


def build() -> tuple[Image.Image, list[dict]]:
    keys = list(PALETTE)
    index = {k: i for i, k in enumerate(keys)}

    # Indexed ("P") image: the file itself guarantees the 32-colour limit.
    # 用索引色（P 模式）影像：檔案格式本身就保證不超過 32 色。
    sheet = Image.new("P", (CELL * COLS, CELL), index["bg"])
    flat = [c for k in keys for c in PALETTE[k]]
    sheet.putpalette(flat + [0, 0, 0] * (256 - len(keys)))

    frames = []
    for i, (name, make) in enumerate(BUYERS):
        sprite = make()
        sprite.check(name)
        ox = i * CELL
        for (x, y), c in sprite.px.items():
            sheet.putpixel((ox + x, y), index[c])
        frames.append({"name": name, "x": ox, "y": 0, "w": CELL, "h": CELL})
    return sheet, frames


def main() -> None:
    sheet, frames = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT_DIR / "buyers.png", optimize=True)
    # NEAREST keeps hard pixel edges; any other filter would blur them.
    # 放大一定要用 NEAREST（最近鄰），其他濾鏡都會把像素邊緣糊掉。
    big = sheet.resize((sheet.width * SCALE, sheet.height * SCALE), Image.NEAREST)
    big.save(OUT_DIR / "buyers@4x.png", optimize=True)
    meta = {
        "image": "buyers.png",
        "cell": CELL,
        "ground_y": GROUND,
        "transparent": "#FF00FF",
        "frames": frames,
    }
    (OUT_DIR / "buyers.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    used = len(set(sheet.tobytes()))
    print(f"wrote {len(frames)} buyers, {used} colours used, to {OUT_DIR}")


if __name__ == "__main__":
    main()
