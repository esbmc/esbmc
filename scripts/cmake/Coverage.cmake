if(ENABLE_COVERAGE)
    if(CMAKE_COMPILER_IS_GNUCXX)        
        if(NOT (CMAKE_BUILD_TYPE STREQUAL "Debug"))
            message(WARNING "Code coverage results with an optimized (non-Debug) build may be misleading")
        endif()

        find_program(LCOV_PATH lcov REQUIRED)
        if(NOT LCOV_PATH)
            message(FATAL_ERROR "lcov not found! Aborting...")
        endif()

        find_program(GENHTML_PATH genhtml)
        if(NOT GENHTML_PATH)
            message(FATAL_ERROR "genhtml not found! Aborting...")
        endif()

       add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
       add_link_options(--coverage)

        
    else()
        message(FATAL_ERROR "Code coverage requires GCC. Aborting.")
    endif()
endif()