add_executable(sell_anything
  src/main.cpp
  src/CanvasScene.cpp
)

target_link_libraries(sell_anything PRIVATE sa::engine)

# Copy runtime DLLs (DGL.dll / DGL_d.dll) next to the executable.
if(WIN32)
  add_custom_command(TARGET sell_anything POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:sell_anything> $<TARGET_FILE_DIR:sell_anything>
    COMMAND_EXPAND_LISTS
  )
endif()
