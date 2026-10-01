cmake_minimum_required(VERSION 3.24)
project(SellAnything VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Backends ----------------------------------------------------------------
# Every backend that is enabled gets compiled in; pick one at run time with --backend.
set(SA_DGL_ROOT "${CMAKE_SOURCE_DIR}/third_party/DGL" CACHE PATH
    "Folder containing DGL's inc/ and lib/x64/ (not redistributable, never committed)")

set(_sa_dgl_found OFF)
if(WIN32 AND EXISTS "${SA_DGL_ROOT}/inc/DGL.h")
  set(_sa_dgl_found ON)
endif()
set(_sa_opengl_default ON)
if(WIN32)
  set(_sa_opengl_default OFF) # Windows uses DGL; the GL backend has no function loader there
endif()

option(SA_BACKEND_DGL "Build the DGL (Direct3D 11) backend" ${_sa_dgl_found})
option(SA_BACKEND_OPENGL "Build the GLFW + OpenGL backend" ${_sa_opengl_default})

if(SA_BACKEND_DGL AND NOT WIN32)
  message(FATAL_ERROR "SA_BACKEND_DGL requires Windows")
endif()
if(SA_BACKEND_DGL AND NOT EXISTS "${SA_DGL_ROOT}/inc/DGL.h")
  message(FATAL_ERROR "DGL not found at SA_DGL_ROOT=${SA_DGL_ROOT} (see docs/backends.md)")
endif()
if(SA_BACKEND_OPENGL AND WIN32)
  message(FATAL_ERROR "SA_BACKEND_OPENGL is not supported on Windows yet; use DGL")
endif()
if(WIN32 AND NOT SA_BACKEND_DGL)
  message(WARNING "DGL not found at ${SA_DGL_ROOT}; only the headless 'null' backend will be built")
endif()

message(STATUS "Sell Anything backends: DGL=${SA_BACKEND_DGL} OpenGL=${SA_BACKEND_OPENGL} null=ON")

# --- Warnings ----------------------------------------------------------------
if(MSVC)
  set(SA_WARNINGS /W4 /permissive-)
else()
  set(SA_WARNINGS -Wall -Wextra -Wpedantic)
endif()

add_subdirectory(engine)
add_subdirectory(game)

target_compile_options(sa_engine PRIVATE ${SA_WARNINGS})
target_compile_options(sell_anything PRIVATE ${SA_WARNINGS})

enable_testing()
add_test(NAME smoke_null_backend COMMAND sell_anything --backend null)
