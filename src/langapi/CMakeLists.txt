add_library(langapi mode.cpp language_ui.cpp language_util.cpp)
target_link_libraries(langapi PUBLIC esbmc_common fmt::fmt filesystem)
