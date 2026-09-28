add_library(abstract-interpretation ai.cpp ai_domain.cpp interval_domain.cpp interval_analysis.cpp gcse.cpp)

target_link_libraries(abstract-interpretation PUBLIC esbmc_common pointeranalysis gotoprograms PRIVATE fmt::fmt)
if(ENABLE_GOTO_CONTRACTOR)
    target_link_libraries(abstract-interpretation PRIVATE gotocontractor)
endif()
