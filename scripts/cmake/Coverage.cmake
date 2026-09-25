include_guard(GLOBAL)

if(ENABLE_COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
        message(FATAL_ERROR
            "ENABLE_COVERAGE requires Clang or AppleClang (found "
            "${CMAKE_CXX_COMPILER_ID}); configure with -DENABLE_COVERAGE=Off "
            "or point CMAKE_CXX_COMPILER at clang++.")
    endif()

    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "Code coverage results with an optimized (non-Debug) build may be misleading")
    endif()

    find_program(LCOV_PATH lcov)
    if(NOT LCOV_PATH)
        message(WARNING "lcov not found — coverage reports will not be filtered (filtered LCOV report will not be produced)")
    endif()

    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate)
endif()
