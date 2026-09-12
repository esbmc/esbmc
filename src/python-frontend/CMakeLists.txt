# `mangle` function is defined in src/scripts/cmake/Utils.cmake
  mangle(${CMAKE_CURRENT_BINARY_DIR}/pythonastgen.c  # an C file output containing python code
         ${CMAKE_CURRENT_SOURCE_DIR} # the input directory folder where python files are located
         RECURSIVE 1
         WILDCARD *.py
         SINGLE
         PREFIX esbmc_pythonastgen_
         MACRO ESBMC_FLAIL
         ACC_HEADERS_INTO ${CMAKE_CURRENT_BINARY_DIR}/pythonastgen.h
         LIST_DIRECTORIES FALSE
  )

add_library(pythonfrontend STATIC
            python_language.cpp
            python_converter.cpp
            converter/converter_binop.cpp
            converter/converter_class.cpp
            converter/converter_compare.cpp
            converter/converter_dunder.cpp
            converter/converter_expr.cpp
            converter/converter_funcall.cpp
            converter/converter_funcdef.cpp
            converter/converter_stmt.cpp
            converter/converter_symbols.cpp
            converter/converter_types.cpp
            converter/converter_unop.cpp
            converter/converter_util.cpp
            param_annotations.cpp
            python_adjust.cpp
            class/python_class.cpp
            class/python_class_builder.cpp
            consteval/python_consteval.cpp
            python-dict/dict_core.cpp
            python-dict/dict_nested.cpp
            python-dict/dict_construction.cpp
            python-dict/dict_access.cpp
            python-dict/dict_type_resolution.cpp
            python-dict/dict_assignment.cpp
            python-dict/dict_methods.cpp
            python-dict/dict_compare.cpp
            python_expr_builder.cpp
            exception/python_exception_handler.cpp
            lambda/python_lambda.cpp
            python-list/list_type_inference.cpp
            python-list/list_construction.cpp
            python-list/list_mutation.cpp
            python-list/list_access.cpp
            python-list/list_query.cpp
            python-list/list_string_ops.cpp
            python-list/list_comprehension.cpp
            python-list/list_set_ops.cpp
            math/python_math.cpp
            set/python_set.cpp
            type/python_typechecking.cpp
            pythonastgen.c
            module/module_manager.cpp
            module/module_locator.cpp
            python_annotation/annotation_intrinsics.cpp
            python_annotation/annotation_parser.cpp
            python_annotation/annotation_utils.cpp
            symbol_id.cpp
            type/element_type_registry.cpp
            type/type_handler.cpp
            math/cmath_lowering_policy.cpp
            math/complex_handler.cpp
            math/complex_handler_utils.cpp
            function_call/builtins.cpp
            function_call/expr.cpp
            function_call/str_conv.cpp
            numpy/numpy_call_expr.cpp
            numpy/ndarray_descriptor.cpp
            function_call/builder.cpp
            string/string_builder.cpp
            string/string_handler.cpp
            string/string_method_handler.cpp
            string/string_handler_utils.cpp
            exception/exception_utils.cpp
            string/char_utils.cpp
            math/math_guard_utils.cpp
            math/parse_float.cpp
            tuple/tuple_handler.cpp
            math/convert_float_literal.cpp
            dynamic_type/dynamic_type_handler.cpp)

target_include_directories(pythonfrontend
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_BINARY_DIR}/src/python-frontend
    PUBLIC ${CMAKE_SOURCE_DIR}/src
    PUBLIC ${Boost_INCLUDE_DIRS}
)

target_link_libraries(pythonfrontend PUBLIC nlohmann_json::nlohmann_json PRIVATE fmt::fmt bigint)

add_executable(python2goto python2goto.cpp)
target_include_directories(python2goto
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${Boost_INCLUDE_DIRS}
)
target_sources(python2goto PRIVATE python_blank_library.cpp)
target_link_libraries(python2goto
  pythonfrontend clangcfrontend clangcppfrontend symex langapi
  solvers clibs gotoalgorithms cache goto2c
  filesystem ${Boost_LIBRARIES} ${OS_INCLUDE_LIBS}
  nlohmann_json::nlohmann_json)

set(pysrc_goto ${CMAKE_CURRENT_BINARY_DIR}/pysrc64.goto)
set(pysrc_c    ${CMAKE_CURRENT_BINARY_DIR}/pysrc64.c)
file(GLOB_RECURSE python_model_files CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/models/*.py")

set(pysrc_entry ${CMAKE_CURRENT_BINARY_DIR}/library_entry.py)
file(WRITE ${pysrc_entry}
     "# Entry module for python2goto: the models are what gets compiled.\n")

add_custom_command(OUTPUT ${pysrc_goto}
  COMMAND python2goto library_entry.py --output ${pysrc_goto}
  DEPENDS python2goto ${python_model_files} ${pysrc_entry}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating Python operational models ${pysrc_goto}"
  VERBATIM)

add_custom_command(OUTPUT ${pysrc_c}
  COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/flail.py
          -o ${pysrc_c} ${pysrc_goto}
  DEPENDS ${pysrc_goto} ${CMAKE_SOURCE_DIR}/scripts/flail.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Converting ${pysrc_goto} to data"
  VERBATIM)

add_library(pysrclib ${pysrc_c} python_library.cpp)
target_include_directories(pysrclib
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_SOURCE_DIR}/src
    PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(pysrclib PRIVATE fmt::fmt)
