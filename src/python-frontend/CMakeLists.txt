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
            pythonastgen.c
            module_manager.cpp
            symbol_id.cpp
            type_handler.cpp
            function_call_expr.cpp
            ${CMAKE_SOURCE_DIR}/src/ansi-c/parse_float.cpp
            ${CMAKE_SOURCE_DIR}/src/ansi-c/convert_float_literal.cpp)

target_include_directories(pythonfrontend
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_BINARY_DIR}/src/python-frontend
    PUBLIC ${Boost_INCLUDE_DIRS}
)

target_link_libraries(pythonfrontend fmt::fmt nlohmann_json::nlohmann_json bigint)
