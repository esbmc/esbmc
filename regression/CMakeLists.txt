# Adds tests as regression
include(CTest)
enable_testing()

find_package(Python)
# TODO: Use add dependency
set(ESBMC_BIN "${CMAKE_BINARY_DIR}/src/esbmc/esbmc")
set(ESBMC_REGRESSION_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/testing_tool.py")

function(add_esbmc_regression folder mode)
    add_test(NAME "regression-${folder}-${mode}"
            COMMAND ${Python_EXECUTABLE} ${ESBMC_REGRESSION_TOOL}
            --tool=${ESBMC_BIN} --regression=${CMAKE_CURRENT_SOURCE_DIR}/${folder} --mode=${mode})
endfunction(add_esbmc_regression mode folder)

set(REGRESSIONS esbmc cstd/ctype cstd/string k-induction llvm digital-filters floats floats-regression)

foreach(regression IN LISTS REGRESSIONS)
    add_esbmc_regression("${regression}" "CORE")
endforeach()