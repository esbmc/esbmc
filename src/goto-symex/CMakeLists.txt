add_library(symex equation/symex_target.cpp equation/symex_target_equation.cpp engine/symex_assign.cpp
  engine/symex_main.cpp  engine/symex_stack.cpp trace/goto_trace.cpp trace/build_goto_trace.cpp
  engine/symex_function.cpp state/goto_symex_state.cpp engine/symex_dereference.cpp engine/symex_goto.cpp
  engine/builtin_functions/memory_alloc.cpp
  engine/builtin_functions/io.cpp
  engine/builtin_functions/cpp_memory.cpp
  engine/builtin_functions/threads.cpp
  engine/builtin_functions/va_arg.cpp
  engine/builtin_functions/memory_ops.cpp
  engine/builtin_functions/object_size.cpp
  engine/builtin_functions/run_builtin.cpp
  engine/builtin_functions/python_builtins.cpp
  engine/builtin_functions/misc.cpp
  engine/builtin_functions/witness.cpp
  equation/slice.cpp engine/symex_other.cpp equation/symex_symmetry.cpp trace/xml_goto_trace.cpp
  engine/symex_valid_object.cpp engine/dynamic_allocation.cpp state/renaming.cpp
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