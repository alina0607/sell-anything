cmake_minimum_required(VERSION 3.24)
project(SellAnything VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Backends ----------------------------------------------------------------
# Every backend that is enabled gets compiled in; pick one at run time with --backend.
# DGL is not redistributable: point at it with -DSA_DGL_ROOT=..., the DGL_ROOT environment
# variable (set it once and every clone finds it), or by copying it into third_party/DGL.
if(DEFINED ENV{DGL_ROOT})
  set(_sa_dgl_default "$ENV{DGL_ROOT}")
else()
  set(_sa_dgl_default "${CMAKE_SOURCE_DIR}/third_party/DGL")
endif()
set(SA_DGL_ROOT "${_sa_dgl_default}" CACHE PATH "Folder containing DGL's inc/ and lib/x64/")

set(_sa_dgl_found OFF)
if(WIN32 AND EXISTS "${SA_DGL_ROOT}/inc/DGL.h")
  set(_sa_dgl_found ON)
endif()

option(SA_BACKEND_OPENGL "Build the GLFW + OpenGL backend" ON)
option(SA_BACKEND_DGL "Build the DGL (Direct3D 11) backend" ${_sa_dgl_found})

if(SA_BACKEND_DGL AND NOT WIN32)
  message(FATAL_ERROR "SA_BACKEND_DGL requires Windows")
endif()
if(SA_BACKEND_DGL AND NOT EXISTS "${SA_DGL_ROOT}/inc/DGL.h")
  message(FATAL_ERROR "DGL not found at SA_DGL_ROOT=${SA_DGL_ROOT} (see docs/backends.md)")
endif()
if(WIN32 AND NOT SA_BACKEND_DGL)
  message(STATUS "DGL not found at ${SA_DGL_ROOT}; building without the DGL backend")
endif()

message(STATUS "Sell Anything backends: OpenGL=${SA_BACKEND_OPENGL} DGL=${SA_BACKEND_DGL} null=ON")

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
