# Adds tests as regression
include(CTest)
enable_testing()

find_package(Python)
# TODO: Use add dependency

set(ESBMC_REGRESSION_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/testing_tool.py")

if(BENCHBRINGUP)
  # To run a specific benchmark in Github workflow and download the log archive when completed
  #   - BENCHMARK_TO_RUN: user-specified benchmark to run
  #   - LOG_DIR: log directory to be archived
  #   - TEST_TIMEOUT: timeout for each test case in a benchmark
  if(NOT DEFINED ENV{BENCHMARK_TO_RUN} OR
     NOT DEFINED ENV{LOG_DIR} OR
     NOT DEFINED ENV{TEST_TIMEOUT})
   message( FATAL_ERROR "Please make sure ENV vars `BENCHMARK_TO_RUN`, `LOG_DIR` and `TEST_TIMEOUT` are defined" )
  endif()

  message(STATUS "Benchmark to run: $ENV{BENCHMARK_TO_RUN} with timeout $ENV{TEST_TIMEOUT}")
  set(BENCHBRINGUP_ARGS --timeout=$ENV{TEST_TIMEOUT} --benchbringup)
endif()

function(add_esbmc_regression_test folder modes test)
    set(test_name "regression/${folder}/${test}")
    add_test(NAME ${test_name}
             COMMAND ${Python_EXECUTABLE} ${ESBMC_REGRESSION_TOOL}
                     --tool=${ESBMC_BIN}
                     --regression=${CMAKE_CURRENT_SOURCE_DIR}/${folder}
                     --modes ${modes}
                     --file=${test}
                     ${BENCHBRINGUP_ARGS}) # additional args for BENCHBRINGUP workflow
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
                    incremental-smt esbmc-cpp11/cpp esbmc-cpp11/cbmc-constructors  extensions)
elseif(WIN32)
    # FUTURE: Add floats-regression esbmc-cpp/cpp
    set(REGRESSIONS esbmc cbmc cstd llvm floats k-induction ${REGRESSIONS_JIMPLE})
else()
  if(NOT BENCHBRINGUP)
    set(REGRESSIONS esbmc-unix esbmc-unix2 esbmc ${REGRESSIONS_SOLIDITY}
                    ${REGRESSIONS_OLD_FRONTEND}
                    cbmc cstd llvm floats floats-regression
                    k-induction esbmc-cpp/cpp esbmc-cpp/cbmc csmith
                    k-induction-parallel nonz3 ${REGRESSIONS_BITWUZLA}
                    incremental-smt esbmc-cpp11/cpp esbmc-cpp11/constructors esbmc-cpp11/new-delete esbmc-cpp11/reference esbmc-cpp/inheritance_bringup
                    esbmc-cpp/polymorphism_bringup
                    esbmc-cpp/bug_fixes
                    ${REGRESSIONS_GOTO_CONTRACTOR} ${REGRESSIONS_JIMPLE} extensions cuda)
  else()
    set(REGRESSIONS $ENV{BENCHMARK_TO_RUN}) # run a single user-specified benchmark
    # create a directory at config time. Test logs will be saved in this directory.
    set(OUT_DIR $ENV{LOG_DIR})
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${OUT_DIR})
  endif()
endif()

foreach(regression IN LISTS REGRESSIONS)
    if(WIN32 OR APPLE) # FUTURE: configure suites using an option
        set(MODES CORE KNOWNBUG FUTURE)
    else()
        set(MODES CORE KNOWNBUG FUTURE THOROUGH)
    endif()
    add_esbmc_regression("${regression}" "${MODES}")
endforeach()


