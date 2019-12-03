
add_library(pointeranalysis value_set.cpp goto_program_dereference.cpp value_set_analysis.cpp dereference.cpp show_value_sets.cpp value_set_domain.cpp)
target_include_directories(pointeranalysis
    PRIVATE ${Boost_INCLUDE_DIRS}
)
target_link_libraries(pointeranalysis util_esbmc)
if (DEFINED HAVE_PYTHON)
  target_link_libraries(pointeranalysis ${PYTHON_LINK_LIBS} dl util expat z pthread)
endif()