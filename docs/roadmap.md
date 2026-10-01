# Roadmap

## Milestone 0 — Project setup
- [x] Repository layout, CMake, CI
- [x] Game ↔ ML protocol stub
- [x] Label all 345 Quick, Draw! categories as `buy` / `refuse`

## Milestone 1 — Engine
- [ ] Window + OpenGL context
- [ ] 2D renderer (lines, quads, text)
- [ ] Mouse input and drawing canvas
- [ ] TCP client talking to the ML service

## Milestone 2 — Sketch classifier
- [ ] Download Quick, Draw! bitmaps
- [ ] Train a CNN baseline; report top-1 / top-3 accuracy
- [ ] Match the game's rasterization to the dataset's
- [ ] Serve it through `classify`

## Milestone 3 — Buyer GPT
- [ ] Train a tokenizer
- [ ] Pre-train on the MiniMind corpus
- [ ] General SFT
- [ ] Generate synthetic buyer dialogues (hold out ~30 unseen categories)
- [ ] Task SFT, then DPO
- [ ] Evaluation: question coverage, repetition, format validity, price error, unseen-category generalization

## Milestone 4 — Playable demo
- [ ] Full round: draw → confirm → negotiate → offer
- [ ] Gameplay video and screenshots in the README
- [ ] Release model weights and dataset on Hugging Face
