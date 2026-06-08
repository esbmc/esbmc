add_library(symex symex_target.cpp symex_target_equation.cpp symex_assign.cpp
  symex_main.cpp  symex_stack.cpp goto_trace.cpp build_goto_trace.cpp
  symex_function.cpp goto_symex_state.cpp symex_dereference.cpp symex_goto.cpp
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
  slice.cpp symex_other.cpp xml_goto_trace.cpp
  symex_valid_object.cpp dynamic_allocation.cpp symex_catch.cpp renaming.cpp
  execution_state.cpp reachability_tree.cpp reachability_tree_cin.cpp
  witnesses.cpp pytest.cpp ctest.cpp printf_formatter.cpp features.cpp html.cpp json.cpp
  sarif.cpp)
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