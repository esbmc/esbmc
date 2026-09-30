add_library(smtfp fp_conv.cpp)
target_link_libraries(smtfp PUBLIC esbmc_common PRIVATE fmt::fmt)
