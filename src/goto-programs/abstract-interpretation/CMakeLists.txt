add_library(abstract-interpretation ai.cpp ai_domain.cpp interval_domain.cpp interval_analysis.cpp dirty.cpp labels.cpp)
target_link_libraries(abstract-interpretation fmt::fmt)