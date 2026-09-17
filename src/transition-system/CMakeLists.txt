add_library(transition-system transition_system.cpp ts_slice.cpp ts_btor2.cpp)
target_link_libraries(transition-system PUBLIC esbmc_common irep2 PRIVATE langapi fmt::fmt)

add_library(transition-system-extraction ts_extraction.cpp)
target_link_libraries(transition-system-extraction PUBLIC esbmc_common transition-system symex)

add_library(transition-system-engines ts_engines.cpp ts_pdr.cpp)
target_link_libraries(transition-system-engines PUBLIC esbmc_common transition-system solvers PRIVATE langapi fmt::fmt)
