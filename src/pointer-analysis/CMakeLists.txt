add_library(pointeranalysis value_set.cpp goto_program_dereference.cpp value_set_analysis.cpp dereference.cpp show_value_sets.cpp value_set_domain.cpp andersen.cpp)
target_link_libraries(pointeranalysis PUBLIC esbmc_common util_esbmc gotoprograms PRIVATE fmt::fmt)
