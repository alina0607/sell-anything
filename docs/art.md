# Character art spec

What to produce so buyers (and later other characters) can be animated in the game. The format
follows the one the CS529 projects already use for the monkey, so the same sheets work in both.

## The reference: the CS529 monkey (measured)

| | Idle | Jump | Walk |
|---|---|---|---|
| File | `MonkeyIdle.png` | `MonkeyJump.png` | `MonkeyWalk.png` |
| Image | 32×32 | 32×32 | 96×96 |
| Grid (`LoadTexture(cols, rows, …)`) | 1×1 | 1×1 | 3×3 |
| Frames (`Play(count, delay, loop)`) | 1, 0 s, no loop | 1, 0 s, no loop | **8** of the 9 cells, 0.05 s (20 fps), loop |
| Mesh UV size | 1 × 1 | 1 × 1 | 1/3 × 1/3 |

- One PNG per animation state; each is a grid of equal cells; frame *i* is at column `i % cols`,
  row `i / cols` (row-major, top-left first); trailing cells may be empty (the walk's 9th).
- Indexed-color (palette) PNG with a transparent color: pixel art.
- Drawn at `Scale 128` from a 32 px cell: an exact 4× enlargement, so pixels stay square.
- The mesh's UV size must equal `1/cols × 1/rows` of the sheet, or frames bleed into each other.

## What the game needs

Buyers sit in the cards on the left of the selling screen. Their animation should follow what the
round already reports for each buyer, so nothing new has to be invented on the server:

| State | When (server event) | Frames | Frame time | Loop |
|---|---|---|---|---|
| `idle` | waiting | 4 | 0.20 s | yes |
| `talk` | asks a question, or a reply line is theirs | 4 | 0.12 s | yes, while their line is newest |
| `happy` | reaction with Δ > 0 | 6 | 0.08 s | once, then `idle` |
| `unhappy` | reaction with Δ < 0 | 6 | 0.08 s | once, then `idle` |
| `leave` | interest hits the walk-away limit | 8 | 0.05 s | once, then hidden |
| `win` | the player sells to them | 6 | 0.08 s | once, hold the last frame |

Per character: **6 PNG sheets**. The 30 buyer names share a smaller set of looks (a name maps to a
look by hash): 8 looks × 6 states = **48 sheets**.

### File format

- **PNG, 8-bit RGBA** (straight alpha, not premultiplied), sRGB. Indexed PNGs with a transparent
  color like the monkey's also load (they are converted to RGBA), but RGBA is the safer default.
- **Cell 64×64 px** (the cards are about 120 px tall; 64 px drawn at 2× = 128 px). Every state of
  every look uses the same cell size and the same **anchor: bottom center (32, 62)**, so states can
  be swapped without the character jumping.
- **Grid**: one row per sheet (`frames × 1`, e.g. 256×64 for 4 frames). That avoids the empty-cell
  bookkeeping; a CS529-style square grid also works if the metadata says how many frames are used.
- **1 px transparent margin inside each cell**, so no frame's pixels touch the next cell.
- **Facing right** (toward the conversation). Mirroring is a negative x scale, so left-facing art isn't needed.
- Pixel art, nearest-neighbor scaling only, integer zoom (1×, 2×, 3×).

### Metadata (one JSON per look)

The numbers `LoadTexture(cols, rows, file)` and `Play(count, delay, loop)` take, in a file instead of code:

```json
{
  "look": "look01",
  "cell": [64, 64],
  "anchor": [32, 62],
  "states": {
    "idle":    { "file": "look01_idle.png",    "cols": 4, "rows": 1, "frames": 4, "frame_time": 0.20, "loop": true },
    "talk":    { "file": "look01_talk.png",    "cols": 4, "rows": 1, "frames": 4, "frame_time": 0.12, "loop": true },
    "happy":   { "file": "look01_happy.png",   "cols": 6, "rows": 1, "frames": 6, "frame_time": 0.08, "loop": false },
    "unhappy": { "file": "look01_unhappy.png", "cols": 6, "rows": 1, "frames": 6, "frame_time": 0.08, "loop": false },
    "leave":   { "file": "look01_leave.png",   "cols": 8, "rows": 1, "frames": 8, "frame_time": 0.05, "loop": false },
    "win":     { "file": "look01_win.png",     "cols": 6, "rows": 1, "frames": 6, "frame_time": 0.08, "loop": false }
  }
}
```

Files: `game/assets/characters/<look>/<look>_<state>.png` + `<look>.json`.

## If the art is generated (not drawn)

Image generators produce one large, soft image at a time and are poor at keeping a character
identical across frames. What works better:

1. Generate **one** clean reference per look (front-ish, facing right, plain background).
2. Turn it into pixel art at the cell size: crop, scale down with nearest neighbor to 64 px tall,
   reduce to a small palette (≤ 32 colors), make the background transparent.
3. Make the frames from that one image: hand edits in a pixel editor (Aseprite, Piskel), or let
   the engine animate a single frame (bob for `idle`, squash for `talk`, a hop for `happy`, a
   shake for `unhappy`, slide out for `leave`). A single frame per state plus engine motion is
   the cheapest version that still reads clearly.

## In the engine

Done (2026-10-04):

- `Backend::LoadTexture(path)` / `TextureSize` / `DrawSprite(texture, UVRect, topLeft, size, alpha)`
  in the OpenGL, DGL and null backends. The null backend really decodes the PNG, so a headless run
  (and CI's end-to-end test) fails on missing or broken art.
- `sa::SpriteSheet` (frame → cell UV, row-major, clamped to the frames in use) and `sa::Animation`
  (`Play(count, frameTime, loop)`, as CS529's `Animation`), unit-tested in `engine/tests/sprite_test.cpp`
  including the monkey's 3×3 / 8-frame walk.
- Pixel art stays sharp: nearest-neighbour sampling and clamped edges (OpenGL), `DGL_TSM_POINT` /
  `DGL_AM_CLAMP` (DGL), drawn at 1× (64 px) in the buyer cards.
- `DrawSprite` takes an alpha, not a tint color: DGL mixes toward a tint where OpenGL would
  multiply, and one meaning across backends matters more than tinting.
- The server reports a `mood` per buyer (docs/protocol.md); `game/src/Characters.cpp` maps it to a
  state, plays reactions once and settles back to idle, walks a leaving buyer off with a fade, and
  picks a look per buyer from a stable hash of the name.

`scripts/make_placeholder_art.py` draws stand-in sheets for 4 looks in exactly this format (each
frame is drawn on its own cell and rejected if it reaches the 1 px margin), so real art replaces
PNGs and nothing else.

## The world and the girl

### Non-negotiable animation consistency

- Never generate or repaint one character keyframe independently and splice it into an existing animation.
  Reject it if palette, value, brush texture, outline, anatomy, head/body ratio, leg length, boot size, or
  rendering detail differs from the approved source set.
- Idle, turn, walk-away, walk-toward, run and jump use one fixed on-screen figure height and one visual palette.
  Moving along the screen's y axis must not scale a character, and changing state must never pop in size.
- Create new motion only by editing or warping pixels from the approved source set, or replace the whole action
  set with one jointly authored set. A separately generated replacement frame is not an acceptable fix.
- Before committing, compare representative frames from every state in one contact sheet at identical height.
  Reject the change if identity, value range, saturation, limb length, costume or rendering style shifts.

Every screen is drawn in one world (`game/assets/stage/`), built from the owner's source art in `art/stage/` by
`python scripts/make_stage_art.py`:

| Source | Becomes | Notes |
|---|---|---|
| `background_strip.png` → `background_tile_crossfade.png` | `background.png` (720 high) | the strip's last 240 px cross-fade into its start, so it repeats without a seam; drawn as many times as the screen needs |
| `stall_cloth.png` (on black) | `stall.png` + `stall.json` | `cloth` is where one draws: the 1240-unit canvas the game keeps strokes in is shown there |
| `cat_shelter.png` | `shelter.png` + `shelter.json` | `easels` are the eight blank canvases, for adoptable cats later |
| `girl_idle.gif` (12 frames on cream) | `girl_idle.png` + `.json` | front view, cut out of the cream |
| `girl_run.gif` + `girl_run_cycle_v2.png` | `girl_run.png` + `.json` | original start/stop transitions plus the jointly authored eight-frame run cycle; every frame is fitted to the same figure height. Never insert a separately generated pose. |
| `girl_walk.gif` (30 frames) | `girl_walk.png` + `.json` | `parts` away 0–11 (up the screen, from behind), turn 12–17, toward 18–29 (down, facing the camera); the turn plays backwards to turn away |
| `npc_<n>.gif` (30 frames) | `npc_<n>.png` + `.json` | a painted buyer: `parts` away 0–11 (walking up, seen from behind), turn 12–17, toward 18–29 (walking off, facing the camera) |
| `Nunito.ttf` (SIL OFL, `fonts/OFL.txt`) | `fonts/nunito.png` + `.txt` | white glyphs at six sizes, drawn in any color |

Painted buyers (`Npc.h`): the n-th buyer of a round is `npc_n` while there are drawings, the rest stay pixel
sprites. Each walks only a short distance up from the flowers at the bottom of the screen to their place in front
of the stall (0.7 s apart, always at the same figure height) and stands there seen from behind; a buyer who loses interest
turns and walks back out through the flowers. Their tag shows only while they stand there, and once they are
gone. Add a buyer by dropping `npc_<n+1>.gif` into `art/stage/` and running the script.

`layout.json` places things and is never overwritten by the script, so it can be tuned by hand: each object's
image, world x of its centre (0 is the stall), the screen y it stands on (`bottom`) and its drawn `height`; the
ground line, the girl's height and start, and the buyers' sprite scale.

The girl (`Girl.h`) runs with ← →, walks up and down the screen with ↑ ↓ (turning round first when needed), and
jumps with Space, except while a text field takes it (the AI field has focus, or an answer is being typed; a
Space typed into an empty answer field is the jump, not a leading space). Every direction and action uses the
same fixed figure height. There are no limits on where she may go yet. The camera stays put while she is in the middle of
the screen and follows her past it. Faces (sheepish, angry, anxious;
see `Stage.h`) are drawn only if `girl_<face>.png` exists; the old ones were in the previous drawing style.

Painted art is sampled smoothly (`TextureFilter::Smooth`), so every sheet cell has a transparent border; the
buyers' pixel art stays crisp (`TextureFilter::Pixel`).
