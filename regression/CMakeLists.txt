# Adds tests as regression
include(CTest)
enable_testing()

find_package(Python)
# TODO: Use add dependency

set(ESBMC_REGRESSION_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/testing_tool.py")

function(add_esbmc_regression_test folder modes test)
    set(test_name "regression/${folder}/${test}")
    add_test(NAME ${test_name}
             COMMAND ${Python_EXECUTABLE} ${ESBMC_REGRESSION_TOOL}
                     --tool=${ESBMC_BIN}
                     --regression=${CMAKE_CURRENT_SOURCE_DIR}/${folder}
                     --modes ${modes}
                     --file=${test})
    set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 10)
endfunction()


function(add_esbmc_regression folder modes)
    SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR}/${regression})
    foreach(test ${SUBDIRS})
        add_esbmc_regression_test(${folder} "${modes}" ${test})
    endforeach()
endfunction()

# conditionally enable regression suites based on configured support
if(ENABLE_BITWUZLA)
    set(REGRESSIONS_BITWUZLA bitwuzla)
endif()
if(ENABLE_SOLIDITY_FRONTEND)
    set(REGRESSIONS_SOLIDITY esbmc-solidity)
endif()
if(ENABLE_GOTO_CONTRACTOR)
    set(REGRESSIONS_GOTO_CONTRACTOR goto-contractor)
endif()
if(ENABLE_OLD_FRONTEND)
    set(REGRESSIONS_OLD_FRONTEND esbmc-old)
endif()
if(ENABLE_JIMPLE_FRONTEND)
    set(REGRESSIONS_JIMPLE jimple)
endif()

# NOTE: In order to make the best of the concurrency set sort the tests from the slowest to fastest.
if(APPLE)
    set(REGRESSIONS esbmc-unix esbmc-unix2 esbmc ${REGRESSIONS_SOLIDITY}
                    cbmc cstd llvm floats floats-regression ${REGRESSIONS_JIMPLE}
                    k-induction k-induction-parallel nonz3 ${REGRESSIONS_BITWUZLA}
                    incremental-smt esbmc-cpp11 extensions)
elseif(WIN32)
    # FUTURE: Add floats-regression esbmc-cpp/cpp
    set(REGRESSIONS esbmc cbmc cstd llvm floats k-induction ${REGRESSIONS_JIMPLE})
else()
    set(REGRESSIONS esbmc-unix esbmc-unix2 esbmc ${REGRESSIONS_SOLIDITY}
                    ${REGRESSIONS_OLD_FRONTEND}
                    cbmc cstd llvm floats floats-regression
                    k-induction esbmc-cpp/cpp esbmc-cpp/cbmc csmith
                    k-induction-parallel nonz3 ${REGRESSIONS_BITWUZLA}
                    incremental-smt esbmc-cpp11/cpp esbmc-cpp11/cbmc-constructors
                    ${REGRESSIONS_GOTO_CONTRACTOR} ${REGRESSIONS_JIMPLE})
endif()

foreach(regression IN LISTS REGRESSIONS)
    if(NOT WIN32) # FUTURE: configure suites using an option
        set(MODES CORE KNOWNBUG FUTURE THOROUGH)
    else()
        set(MODES CORE KNOWNBUG FUTURE)
    endif()
    add_esbmc_regression("${regression}" "${MODES}")
endforeach()
