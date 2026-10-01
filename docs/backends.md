# Rendering backends

The engine talks to the platform through one interface, [`sa::Backend`](../engine/include/sa/Backend.h):
a window, mouse/keyboard input, and immediate-mode 2D drawing (`DrawLine`, `DrawRect`).
Game code never includes a graphics API header, so the same game runs on every backend.

| Backend | `--backend` | Platform | Graphics API | Built by default |
|---|---|---|---|---|
| OpenGL | `opengl` | Windows, macOS, Linux | OpenGL 3.3 core via GLFW + glad | yes (default) |
| DGL | `dgl` | Windows | Direct3D 11 (via DigiPen Graphics Library) | when DGL is found ([below](#using-dgl-on-windows)) |
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

- **OpenGL**: functions are loaded with glad (bundled in GLFW's sources), so the same code runs on
  every OS. The vertex shader maps pixels to NDC; the viewport uses the framebuffer size so HiDPI
  displays render at full resolution.
- **DGL**: points go through `DGL_Camera_ScreenCoordToWorld`, since DGL draws in a centered, y-up world space.

## Building on Windows

A fresh clone builds and runs the **OpenGL** backend with no setup (x64, Visual Studio 2022 or newer):

```bat
cmake -S . -B build -A x64
cmake --build build --config Release
build\game\Release\sell_anything.exe
```

Visual Studio users can also open `build\SellAnything.sln`.

## Using DGL on Windows

DGL is © DigiPen and **must not be committed** to this repository, so it needs a one-time setup.
CMake looks for a folder containing `inc/DGL.h` and `lib/x64/` (`DGL.lib`, `DGL.dll`, `DGL_d.lib`, `DGL_d.dll`),
checking these in order:

| | How | When to use |
|---|---|---|
| 1 | `cmake ... -DSA_DGL_ROOT=C:/path/to/DGL` | one-off |
| 2 | `DGL_ROOT` environment variable | **recommended**: set once, every clone finds it |
| 3 | copy the folder to `third_party/DGL/` (git-ignored) | per clone |

Any CS529 project's `Libraries/DGL` folder works. To set the environment variable once
(then open a new terminal):

```powershell
setx DGL_ROOT "C:\path\to\CS529\Project3\Libraries\DGL"
```

Then build as above and run:

```bat
build\game\Release\sell_anything.exe --backend dgl
```

The build copies `DGL.dll` (or `DGL_d.dll` for Debug) next to the executable.
If DGL isn't found, the build still succeeds with OpenGL and null; CMake's output says which backends were built.

## Why OpenGL + DGL, and not Vulkan (yet)

| | OpenGL | DGL | Vulkan |
|---|---|---|---|
| Windows | ✅ | ✅ the course's own library | ✅ |
| macOS (where development happens) | ✅ | ❌ | ⚠️ only through MoltenVK |
| Code to draw the first line | ~150 lines | ~30 lines | ~1,000+ lines |
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
