# Roadmap

## Milestone 0 — Project setup
- [x] Repository layout, CMake, CI
- [x] Game ↔ ML protocol stub
- [x] Label all 345 Quick, Draw! categories as `buy` / `refuse`

## Milestone 1 — Engine
- [x] Backend interface, selectable at run time (`--backend`)
- [x] OpenGL backend (GLFW, macOS/Linux), batched into one draw call per frame
- [x] DGL backend (Direct3D 11, Windows) — compiles against DGL 1.4.0; needs a run on Windows
- [x] Headless backend + CI smoke test
- [x] Mouse input and drawing canvas
- [ ] Text rendering (bitmap font) for the dialogue UI
- [ ] Text input for the player's answers
- [ ] Render the canvas to a texture instead of redrawing every segment each frame
- [ ] TCP client talking to the ML service
- [ ] *Stretch:* Vulkan backend

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
