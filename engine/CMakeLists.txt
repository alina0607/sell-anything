add_library(sa_engine
  src/Engine.cpp
  src/BackendRegistry.cpp
)
add_library(sa::engine ALIAS sa_engine)

target_include_directories(sa_engine
  PUBLIC include
  PRIVATE src
)

# --- OpenGL backend (GLFW) ---------------------------------------------------
if(SA_BACKEND_OPENGL)
  include(FetchContent)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(glfw)

  target_sources(sa_engine PRIVATE src/backends/OpenGLBackend.cpp)
  # GLFW loads the system OpenGL library at run time; glad (bundled with GLFW) loads the functions.
  target_include_directories(sa_engine PRIVATE "${glfw_SOURCE_DIR}/deps")
  target_link_libraries(sa_engine PRIVATE glfw)
  target_compile_definitions(sa_engine PRIVATE SA_HAS_OPENGL=1)
endif()

# --- DGL backend (Windows, local SDK only) -----------------------------------
if(SA_BACKEND_DGL)
  add_library(DGL::DGL SHARED IMPORTED GLOBAL)
  set_target_properties(DGL::DGL PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SA_DGL_ROOT}/inc"
    IMPORTED_CONFIGURATIONS "DEBUG;RELEASE"
    IMPORTED_IMPLIB_DEBUG "${SA_DGL_ROOT}/lib/x64/DGL_d.lib"
    IMPORTED_LOCATION_DEBUG "${SA_DGL_ROOT}/lib/x64/DGL_d.dll"
    IMPORTED_IMPLIB_RELEASE "${SA_DGL_ROOT}/lib/x64/DGL.lib"
    IMPORTED_LOCATION_RELEASE "${SA_DGL_ROOT}/lib/x64/DGL.dll"
    MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE
    MAP_IMPORTED_CONFIG_MINSIZEREL RELEASE
  )

  target_sources(sa_engine PRIVATE src/backends/DGLBackend.cpp)
  target_link_libraries(sa_engine PUBLIC DGL::DGL)
  target_compile_definitions(sa_engine PRIVATE SA_HAS_DGL=1 NOMINMAX)
endif()
