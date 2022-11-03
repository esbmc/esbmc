add_library(gotoprograms goto_convert.cpp goto_function.cpp goto_main.cpp
  goto_sideeffects.cpp goto_program.cpp goto_check.cpp goto_inline.cpp
  remove_skip.cpp goto_convert_functions.cpp remove_unreachable.cpp
  builtin_functions.cpp show_claims.cpp destructor.cpp set_claims.cpp
  add_race_assertions.cpp rw_set.cpp read_goto_binary.cpp static_analysis.cpp
  goto_program_serialization.cpp goto_function_serialization.cpp
  read_bin_goto_object.cpp goto_program_irep.cpp format_strings.cpp
  loop_numbers.cpp goto_loops.cpp write_goto_binary.cpp
  goto_k_induction.cpp loopst.cpp ai.cpp ai_domain.cpp interval_analysis.cpp interval_domain.cpp)
add_library(gotoalgorithms goto_slicer.cpp loop_unroll.cpp mark_decl_as_non_det.cpp)

if(ENABLE_GOTO_CONTRACTOR)
    include(FindIbex)
    add_library(gotocontractor goto_contractor.cpp)
    target_compile_options(gotocontractor PUBLIC -I${IBEX_INCLUDE_DIRS}/ibex/3rd -I${IBEX_INCLUDE_DIRS} -I${IBEX_INCLUDE_DIRS}/ibex )
    target_include_directories(gotocontractor PUBLIC ${IBEX_INCLUDE_DIRS} ${IBEX_INCLUDE_DIRS}/ibex ${IBEX_INCLUDE_DIRS}/ibex/3rd)
    target_link_libraries(gotocontractor PUBLIC algorithms gotoprograms ${IBEX_LIB} ${IBEX_LIB_GAOL} ${IBEX_LIB_GDTOA} ${IBEX_LIB_ULTIM} ${IBEX_LIB_SOPLEX} -lz)
endif()

target_link_libraries(gotoalgorithms algorithms gotoprograms)
target_include_directories(gotoprograms
    PRIVATE ${Boost_INCLUDE_DIRS}
)
target_include_directories(gotoalgorithms
    PRIVATE ${Boost_INCLUDE_DIRS}
)

target_link_libraries(gotoprograms pointeranalysis bigint)
