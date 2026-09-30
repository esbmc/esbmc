add_library(transition-system transition_system.cpp ts_slice.cpp)
target_link_libraries(transition-system PUBLIC esbmc_common irep2 PRIVATE langapi fmt::fmt)

add_library(transition-system-extraction ts_extraction.cpp)
target_link_libraries(transition-system-extraction PUBLIC esbmc_common transition-system symex)
