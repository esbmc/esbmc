add_library(llm
  llm_client.cpp
  cli_llm_client.cpp
)

if(Boost_VERSION VERSION_GREATER_EQUAL 1.88)
  find_package(Boost REQUIRED COMPONENTS process)
  target_link_libraries(llm PRIVATE Boost::process)
endif()

target_link_libraries(llm
  PUBLIC util_esbmc
  PRIVATE ESBMC::boost fmt::fmt
)
