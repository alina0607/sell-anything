add_library(sa_engine
  src/Engine.cpp
)
add_library(sa::engine ALIAS sa_engine)

target_include_directories(sa_engine PUBLIC include)
