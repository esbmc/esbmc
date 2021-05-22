# Apple complains about cmdline.isset and windows complains about boost
if(UNIX AND NOT APPLE)
  add_library(test_util_irep util_irep.cpp)
  target_link_libraries(test_util_irep ${Boost_LIBRARIES} bigint util_esbmc)

  add_library(test_goto_factory goto_factory.cpp)  
  target_link_libraries(test_goto_factory clangcfrontend clangcppfrontend clibs)
endif()
