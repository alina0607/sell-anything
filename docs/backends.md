# Rendering backends

The engine talks to the platform through one interface, [`sa::Backend`](../engine/include/sa/Backend.h):
a window, mouse/keyboard input, and immediate-mode 2D drawing (`DrawLine`, `DrawRect`).
Game code never includes a graphics API header, so the same game runs on every backend.

| Backend | `--backend` | Platform | Graphics API | Built by default |
|---|---|---|---|---|
| DGL | `dgl` | Windows | Direct3D 11 (via DigiPen Graphics Library) | when `third_party/DGL` exists |
| OpenGL | `opengl` | macOS, Linux | OpenGL 3.3 core via GLFW | yes |
| Null | `null` | all | none (headless) | always |

Every enabled backend is compiled in; choose one at run time:

```bash
sell_anything --list-backends
sell_anything --backend dgl
```

With no flag, the first available backend in the table is used.

## Coordinate convention

All backends take **window pixels, origin top-left, y down** — the same convention as the
Quick, Draw! stroke format, so strokes go to the classifier without conversion.
Each backend converts internally:

- **OpenGL**: the vertex shader maps pixels to NDC; the viewport uses the framebuffer size so HiDPI displays render at full resolution.
- **DGL**: points go through `DGL_Camera_ScreenCoordToWorld`, since DGL draws in a centered, y-up world space.

## Using DGL on Windows

DGL is © DigiPen and **must not be committed** to this repository. `third_party/DGL/` is git-ignored.

1. Copy the `DGL` folder from any CS529 project (`Libraries/DGL`) to `third_party/DGL`, so that these exist:
   ```
   third_party/DGL/inc/DGL.h
   third_party/DGL/lib/x64/DGL.lib   DGL.dll   DGL_d.lib   DGL_d.dll
   ```
   Or point CMake somewhere else with `-DSA_DGL_ROOT=C:/path/to/DGL`.
2. Configure and build (x64 only — DGL ships no 32-bit binaries):
   ```bat
   cmake -S . -B build -A x64
   cmake --build build --config Release
   build\game\Release\sell_anything.exe --backend dgl
   ```
   The build copies `DGL.dll` (or `DGL_d.dll` for Debug) next to the executable.
   Visual Studio users can open `build\SellAnything.sln` directly.

If `third_party/DGL` is missing, Windows builds still succeed with only the `null` backend and CMake prints a warning.

## Why DGL + OpenGL, and not Vulkan (yet)

| | DGL | OpenGL | Vulkan |
|---|---|---|---|
| Runs on the Windows demo machines | ✅ the course's own library | ⚠️ needs a function loader | ✅ |
| Runs on macOS (where development happens) | ❌ | ✅ native | ⚠️ only through MoltenVK |
| Code to draw the first line | ~30 lines | ~150 lines | ~1,000+ lines |
| What this game needs | colored lines and quads | colored lines and quads | — |

The game draws strokes and UI panels: a few thousand vertices per frame. Vulkan's strengths
(explicit synchronization, multithreaded command recording) don't pay off at that scale,
and it would roughly double the engine code without changing what the player sees.

The interesting engineering is the **backend boundary itself**. Adding Vulkan later means
writing one more `Backend` subclass; no game code changes. That stays on the roadmap as a stretch goal.

## Adding a backend

1. Implement `sa::Backend` in `engine/src/backends/`.
2. Add a CMake option and `SA_HAS_<NAME>` define in `engine/CMakeLists.txt`.
3. Register it in `engine/src/BackendRegistry.cpp`.
