# Sell Anything

> Draw anything. Try to sell it. An AI buyer decides what it's worth.

**Sell Anything** is a game built end-to-end from scratch:

- a **custom C++ game engine** (rendering, input, game loop),
- a **sketch classifier** trained on Google's [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) dataset (PyTorch),
- a **small GPT trained from scratch** that role-plays a buyer: it asks about the item, then makes an offer — or refuses to buy things that can't (or shouldn't) be sold.

> 🚧 **Status:** early development. See the [roadmap](docs/roadmap.md).

---

## How a round plays

```
 ┌────────────┐   strokes    ┌──────────────────┐   "bicycle"   ┌──────────────────┐
 │  Player    │ ───────────▶ │ Sketch classifier│ ────────────▶ │  Buyer GPT       │
 │  draws     │              │  (CNN, 345 cls)  │               │  asks questions, │
 │  with mouse│ ◀─────────── │                  │               │  makes an offer  │
 └────────────┘  "Is this a  └──────────────────┘               └──────────────────┘
                  bicycle?"
```

```
Buyer:  Nice bike! How many years have you had it?
Player: About two years.
Buyer:  Any rust or damage to the frame?
Player: A small scratch on the frame, otherwise fine.
Buyer:  [OFFER] NT$2,800
        [REASON] Two years old with a cosmetic scratch — a bit under typical resale.
```

Try to sell a tiger, a rainbow, or your elbow and the buyer will politely refuse.

## Architecture

| Component | Language | Path | Description |
|---|---|---|---|
| Engine | C++20 | [`engine/`](engine) | Game loop, rendering, input, scenes |
| Game | C++20 | [`game/`](game) | Drawing canvas, dialogue UI, game states |
| ML service | Python | [`ml/`](ml) | Sketch classifier + buyer GPT behind a local socket |

The game and the ML service are separate processes that talk over **newline-delimited JSON on localhost TCP**.
That keeps the engine free of Python dependencies and lets each side be developed and tested on its own.
See [`docs/architecture.md`](docs/architecture.md) and the [protocol spec](docs/protocol.md).

## Building

### Game (C++)

Requires CMake ≥ 3.24 and a C++20 compiler.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/game/sell_anything
```

### ML service (Python)

Requires Python ≥ 3.10.

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m sellanything_ml.server        # listens on 127.0.0.1:5555
pytest
```

## Data

| Dataset | Used for |
|---|---|
| [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) (CC BY 4.0) | Sketch classifier, item list |
| [MiniMind dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset) | GPT pre-training and general SFT |
| Synthetic buyer dialogues (generated) | Task-specific SFT / DPO |

Each of the 345 Quick, Draw! categories is labeled `buy` or `refuse` in [`ml/data/`](ml/data).

## Acknowledgements

- [nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- [MiniMind](https://github.com/jingyaogong/minimind) by jingyaogong
- [Quick, Draw!](https://quickdraw.withgoogle.com/data) by Google Creative Lab

## License

[MIT](LICENSE)
