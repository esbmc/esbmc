add_library(symex equation/symex_target.cpp equation/symex_target_equation.cpp symex_assign.cpp
  symex_main.cpp  symex_stack.cpp trace/goto_trace.cpp trace/build_goto_trace.cpp
  symex_function.cpp state/goto_symex_state.cpp symex_dereference.cpp symex_goto.cpp
  builtin_functions/memory_alloc.cpp
  builtin_functions/io.cpp
  builtin_functions/cpp_memory.cpp
  builtin_functions/threads.cpp
  builtin_functions/va_arg.cpp
  builtin_functions/memory_ops.cpp
  builtin_functions/object_size.cpp
  builtin_functions/run_builtin.cpp
  builtin_functions/python_builtins.cpp
  builtin_functions/misc.cpp
  builtin_functions/witness.cpp
  equation/slice.cpp symex_other.cpp equation/symex_symmetry.cpp trace/xml_goto_trace.cpp
  symex_valid_object.cpp dynamic_allocation.cpp state/renaming.cpp
  scheduler/execution_state.cpp scheduler/reachability_tree.cpp scheduler/reachability_tree_cin.cpp
  symex_invariant.cpp
  witness/witnesses.cpp testgen/pytest.cpp testgen/ctest.cpp trace/printf_formatter.cpp equation/features.cpp trace/html.cpp trace/json.cpp
  trace/sarif.cpp)
target_include_directories(symex
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${Boost_INCLUDE_DIRS}
)
target_compile_definitions(symex PUBLIC BOOST_ALL_NO_LIB YAML_CPP_STATIC_DEFINE)
target_link_libraries(symex
  PUBLIC  smt gotoprograms pointeranalysis abstract-interpretation
  PRIVATE nlohmann_json::nlohmann_json yaml-cpp::yaml-cpp fmt::fmt usr_utils
)

if(WIN32)
  target_link_libraries(symex PRIVATE nlohmann_json::nlohmann_json yaml-cpp::yaml-cpp fmt::fmt wsock32 ws2_32)
endif()