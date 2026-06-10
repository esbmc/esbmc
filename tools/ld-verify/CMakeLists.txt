add_executable(ld-verify main.cpp)

target_include_directories(ld-verify
  PRIVATE ${CMAKE_SOURCE_DIR}/src
  PRIVATE ${CMAKE_BINARY_DIR}/src
)

# Link only the LD frontend and core ESBMC libraries needed for parse + show_parse.
# Full verification (typecheck + symex) requires linking the full ESBMC driver,
# which is handled by invoking the esbmc binary rather than linking it here.
target_link_libraries(ld-verify
  PRIVATE ldfrontend util_esbmc irep2 langapi
          ${Boost_LIBRARIES}
)
