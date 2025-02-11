add_library(abstract-interpretation ai.cpp ai_domain.cpp interval_domain.cpp interval_analysis.cpp gcse.cpp)
target_include_directories(abstract-interpretation
        PUBLIC ${Boost_INCLUDE_DIRS})

if (ENABLE_GOTO_CONTRACTOR)
    target_link_libraries(abstract-interpretation gotocontractor pointeranalysis fmt::fmt)
else ()
    target_link_libraries(abstract-interpretation pointeranalysis fmt::fmt)
endif ()
