# Adds tests as regression
include(CTest)
enable_testing()

find_package(Python)
# TODO: Use add dependency

set(ESBMC_REGRESSION_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/testing_tool.py")

function(add_esbmc_regression folder mode)
    add_test(NAME "regression-${folder}-${mode}"
            COMMAND ${Python_EXECUTABLE} ${ESBMC_REGRESSION_TOOL}
            --tool=${ESBMC_BIN} --regression=${CMAKE_CURRENT_SOURCE_DIR}/${folder} --mode=${mode})
endfunction(add_esbmc_regression mode folder)

if(APPLE)
    set(REGRESSIONS esbmc cstd llvm floats floats-regression k-induction esbmc-unix)
else()
    set(REGRESSIONS esbmc cstd llvm floats floats-regression k-induction esbmc-cpp/cpp csmith esbmc-unix)
endif()

foreach(regression IN LISTS REGRESSIONS)
    add_esbmc_regression("${regression}" "CORE")
    add_esbmc_regression("${regression}" "KNOWNBUG")
    add_esbmc_regression("${regression}" "FUTURE")

    if(NOT APPLE)        
        add_esbmc_regression("${regression}" "THOROUGH")        
    endif()
endforeach()