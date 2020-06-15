# Adds tests as regression
include(CTest)
enable_testing()

find_package(Python)
# TODO: Use add dependency

set(ESBMC_BIN "${CMAKE_BINARY_DIR}/src/esbmc/esbmc")
if(APPLE)
    set(MACOS_ESBMC_WRAPPER "#!/bin/sh\n${ESBMC_BIN} -I${C2GOTO_INCLUDE_DIR} $@")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/macos-wrapper.sh ${MACOS_ESBMC_WRAPPER})
    set(ESBMC_BIN "${CMAKE_CURRENT_BINARY_DIR}/macos-wrapper.sh")
endif()

set(ESBMC_REGRESSION_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/testing_tool.py")

function(add_esbmc_regression folder mode)
    add_test(NAME "regression-${folder}-${mode}"
            COMMAND ${Python_EXECUTABLE} ${ESBMC_REGRESSION_TOOL}
            --tool=${ESBMC_BIN} --regression=${CMAKE_CURRENT_SOURCE_DIR}/${folder} --mode=${mode} --library=${CMAKE_SOURCE_DIR}/src/cpp/library)
endfunction(add_esbmc_regression mode folder)

# TODO: Do not merge into master, this is intended only for clang-cpp-branch
set(REGRESSIONS esbmc-cpp/cpp)
#set(REGRESSIONS esbmc cstd llvm floats floats-regression k-induction esbmc-cpp/cpp)

foreach(regression IN LISTS REGRESSIONS)
    add_esbmc_regression("${regression}" "CORE")
    add_esbmc_regression("${regression}" "KNOWNBUG")
    add_esbmc_regression("${regression}" "FUTURE")

    if(NOT APPLE)        
        add_esbmc_regression("${regression}" "THOROUGH")        
    endif()
endforeach()
