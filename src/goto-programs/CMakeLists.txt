add_library(gotoprograms goto_convert.cpp goto_function.cpp goto_main.cpp goto_sideeffects.cpp goto_program.cpp goto_check.cpp goto_inline.cpp remove_skip.cpp goto_convert_functions.cpp remove_unreachable.cpp builtin_functions.cpp show_claims.cpp destructor.cpp set_claims.cpp add_race_assertions.cpp rw_set.cpp read_goto_binary.cpp static_analysis.cpp goto_program_serialization.cpp goto_function_serialization.cpp read_bin_goto_object.cpp goto_program_irep.cpp format_strings.cpp loop_numbers.cpp goto_loops.cpp write_goto_binary.cpp goto_unwind.cpp goto_k_induction.cpp loopst.cpp goto_python.cpp ai.cpp ai_domain.cpp interval_analysis.cpp interval_domain.cpp)
target_include_directories(gotoprograms
    PRIVATE ${Boost_INCLUDE_DIRS}
)
target_link_libraries(gotoprograms pointeranalysis util_esbmc bigint)
if (DEFINED HAVE_PYTHON)
  target_link_libraries(gotoprograms ${PYTHON_LINK_LIBS} dl util expat z pthread)
endif()