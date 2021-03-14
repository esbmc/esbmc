# Adds tests as regression
include(CTest)
enable_testing()

find_package(Python)
# TODO: Use add dependency

set(ESBMC_REGRESSION_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/testing_tool.py")

function(add_esbmc_regression_test folder modes test)
    set(test_name "regression-${folder}-${test}")
    add_test(NAME ${test_name}
            COMMAND ${Python_EXECUTABLE} ${ESBMC_REGRESSION_TOOL}
            --tool=${ESBMC_BIN} --regression=${CMAKE_CURRENT_SOURCE_DIR}/${folder} --modes ${modes} --file=${test})
    set_tests_properties(${test_name} PROPERTIES
            SKIP_RETURN_CODE 10)
endfunction()


function(add_esbmc_regression folder modes)
    SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR}/${regression})
    foreach(test ${SUBDIRS})
        add_esbmc_regression_test(${folder} "${modes}" ${test})
    endforeach()
endfunction()

if(APPLE)
    set(REGRESSIONS esbmc cbmc cstd llvm floats floats-regression k-induction esbmc-unix esbmc-unix2 k-induction-parallel nonz3)
elseif(WIN32)
    # FUTURE: Add floats-regression esbmc-cpp/cpp
    set(REGRESSIONS esbmc cbmc cstd llvm floats  k-induction )
else()
    set(REGRESSIONS esbmc cbmc cstd llvm floats floats-regression k-induction esbmc-cpp/cpp csmith esbmc-unix esbmc-unix2 k-induction-parallel nonz3)
endif()

set(REGRESSIONS esbmc-cpp/cpp)
foreach(regression IN LISTS REGRESSIONS)
    if(ENABLE_YICES AND ENABLE_CVC4) # FUTURE: configure suites using an option
        set(MODES CORE KNOWNBUG FUTURE THOROUGH)
    else()
        set(MODES CORE KNOWNBUG FUTURE)
    endif()
    add_esbmc_regression("${regression}" "${MODES}")
endforeach()
