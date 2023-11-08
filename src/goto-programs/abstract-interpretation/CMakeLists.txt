add_library(abstract-interpretation ai.cpp ai_domain.cpp interval_domain.cpp interval_analysis.cpp)


if (ENABLE_GOTO_CONTRACTOR)
    #include(FindIbex)
    target_link_libraries(abstract-interpretation fmt::fmt PkgConfig::IBEX)
    target_include_directories(abstract-interpretation
            PUBLIC ${Boost_INCLUDE_DIRS} ${IBEX_INCLUDE_DIRS} ${IBEX_INCLUDE_DIRS}/ibex ${IBEX_INCLUDE_DIRS}/ibex/3rd
    )
else ()
    target_link_libraries(abstract-interpretation fmt::fmt)
    target_include_directories(abstract-interpretation
            PUBLIC ${Boost_INCLUDE_DIRS})
endif ()