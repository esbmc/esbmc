add_library(llm_goto
  render.cpp
  parse_expr.cpp
)

target_link_libraries(llm_goto
  PUBLIC gotoprograms util_esbmc
  PRIVATE fmt::fmt
)
