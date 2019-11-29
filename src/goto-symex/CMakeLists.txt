add_library(symex symex_python.cpp symex_target.cpp symex_target_equation.cpp symex_assign.cpp symex_main.cpp goto_trace.cpp build_goto_trace.cpp symex_function.cpp goto_symex_state.cpp symex_dereference.cpp symex_goto.cpp builtin_functions.cpp slice.cpp symex_other.cpp xml_goto_trace.cpp symex_valid_object.cpp dynamic_allocation.cpp symex_catch.cpp renaming.cpp execution_state.cpp reachability_tree.cpp witnesses.cpp printf_formatter.cpp)
target_include_directories(symex 
    PRIVATE ${CMAKE_BINARY_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(symex pointeranalysis langapi util_esbmc bigint)
if (DEFINED HAVE_PYTHON)
  target_link_libraries(symex -lm -ldl ${PYTHON_LINK_LIBS}  dl util expat z pthread)
endif()