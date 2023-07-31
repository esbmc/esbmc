add_library(abstract-interpretation ai.cpp ai_domain.cpp interval_domain.cpp interval_analysis.cpp common_subexpression_elimination.cpp)
target_compile_features(abstract-interpretation PRIVATE cxx_std_20)
target_include_directories(abstract-interpretation
        PUBLIC ${Boost_INCLUDE_DIRS})

if (ENABLE_GOTO_CONTRACTOR)
    target_link_libraries(abstract-interpretation gotocontractor fmt::fmt)
else ()
    target_link_libraries(abstract-interpretation fmt::fmt)
endif ()
