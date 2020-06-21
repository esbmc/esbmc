# Module to setup and configure clang tidy for ESBMC

if(ENABLE_CLANG_TIDY)
    message(STATUS "Clang-tidy enabled")
    find_program(CLANG_TIDY clang-tidy)
    if(NOT CLANG_TIDY)
        message(FATAL_ERROR "clang-tidy not found!")
    endif()
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY} -header-filter=.,-checks=-*,readability-*,-warnings-as-errors=*)
endif()