add_executable(sell_anything
  src/main.cpp
)

target_link_libraries(sell_anything PRIVATE sa::engine)
