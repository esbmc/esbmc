add_library(abstract-interpretation ai.cpp ai_domain.cpp interval_domain.cpp interval_analysis.cpp)
target_link_libraries(abstract-interpretation fmt::fmt)
target_include_directories(abstract-interpretation
        PUBLIC ${Boost_INCLUDE_DIRS}
        )