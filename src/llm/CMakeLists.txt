add_library(llm
  llm_client.cpp
  cli_llm_client.cpp
  openai_llm_client.cpp
)

target_link_libraries(llm
  PUBLIC util_esbmc
  PRIVATE ESBMC::boost fmt::fmt
)
