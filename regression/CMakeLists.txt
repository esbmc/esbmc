# Adds tests as regression
include(CTest)
enable_testing()

# Use Python3_EXECUTABLE if set (from build.sh), otherwise find Python
if(DEFINED Python3_EXECUTABLE)
    set(Python_EXECUTABLE ${Python3_EXECUTABLE})
    message(STATUS "Using Python3_EXECUTABLE: ${Python_EXECUTABLE}")
else()
    find_package(Python)
    # TODO: Use add dependency
endif()

set(ESBMC_REGRESSION_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/testing_tool.py")

# Per-test timeout, injected into the runner via
# the ESBMC_REGRESS_TIMEOUT environment variable
#so the Python side never needs its own default.
set(ESBMC_REGRESS_TIMEOUT 1200 CACHE STRING
    "Per-test timeout in seconds for regression tests")
set(ESBMC_REGRESS_MEMORY_LIMIT 8192 CACHE STRING
    "Per-test virtual memory limit")

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

  # Override the cache value with the CI-supplied one.
  set(ESBMC_REGRESS_TIMEOUT $ENV{TEST_TIMEOUT})
  message(STATUS "Benchmark to run: $ENV{BENCHMARK_TO_RUN} with timeout ${ESBMC_REGRESS_TIMEOUT}")
  set(BENCHBRINGUP_ARGS --benchbringup)
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
    # Timeout and memory limit are passed via environment
    set_tests_properties(${test_name}
      PROPERTIES SKIP_RETURN_CODE 10
      TIMEOUT ${ESBMC_REGRESS_TIMEOUT}
      ENVIRONMENT "ESBMC_REGRESS_TIMEOUT=${ESBMC_REGRESS_TIMEOUT};ESBMC_REGRESS_MEMORY_LIMIT=${ESBMC_REGRESS_MEMORY_LIMIT}"
      LABELS "regression;${folder}/")
endfunction()

function(add_esbmc_regression folder modes)
    subdirlist(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR}/${regression})
    if(${folder} STREQUAL "quixbugs")
        list(REMOVE_ITEM SUBDIRS bucketsort)
    endif()
    foreach(test ${SUBDIRS})
        add_esbmc_regression_test(${folder} "${modes}" ${test})
    endforeach()
endfunction()

# conditionally enable regression suites based on configured support
if(ENABLE_BITWUZLA)
    set(REGRESSIONS_BITWUZLA bitwuzla)
endif()
if(ENABLE_CVC4)
    set(REGRESSIONS_CVC cvc)
endif()
if(ENABLE_CVC5)
    set(REGRESSIONS_CVC cvc)
endif()
if(ENABLE_MATHSAT)
    set(REGRESSIONS_MATHSAT mathsat)
endif()
if(ENABLE_SMTLIB)
    set(REGRESSIONS_SMTLIB smtlib)
endif()
if(ENABLE_Z3)
    set(REGRESSIONS_Z3 z3 ir-ra)
endif()
if(ENABLE_SOLIDITY_FRONTEND)
    set(REGRESSIONS_SOLIDITY esbmc-solidity)
endif()
if(ENABLE_GOTO_CONTRACTOR)
    set(REGRESSIONS_GOTO_CONTRACTOR Interval-analysis-ibex-contractor)
endif()
if(ENABLE_JIMPLE_FRONTEND)
    set(REGRESSIONS_JIMPLE jimple)
endif()
if(ENABLE_PYTHON_FRONTEND)
    set(REGRESSIONS_PYTHON python)
    set(REGRESSIONS_PYTHON_INTENSIVE python-intensive)
    set(REGRESSIONS_MOPSA mopsa)
    set(REGRESSIONS_NUMPY numpy)
    set(REGRESSIONS_HUMANEVAL humaneval)
    set(REGRESSIONS_QUIXBUGS quixbugs)
    set(REGRESSIONS_PYTHON_COVERAGE python-coverage)
    set(REGRESSIONS_AI_GENERATED_CODE ai-generated-code)
endif()

if(ESBMC_CHERI_CLANG)
    set(REGRESSIONS_CHERI cheri-c
                          cheri-128
       )
endif()

# list of C++98/03 test suites
set(REGRESSIONS_CPP03 esbmc-cpp/cpp
                      esbmc-cpp/cbmc
                      esbmc-cpp/destructors
                      esbmc-cpp/polymorphism_bringup
                      esbmc-cpp/polymorphism_bringup_overload
                      esbmc-cpp/inheritance
                      esbmc-cpp/inheritance_bringup
                      esbmc-cpp/bug_fixes
                      esbmc-cpp/OM_sanity_checks
                      esbmc-cpp/string
                      esbmc-cpp/stream
                      esbmc-cpp/gcc-template-tests
                      esbmc-cpp/template
                      esbmc-cpp/algorithm
                      esbmc-cpp/deque
                      esbmc-cpp/list
                      esbmc-cpp/map
                      esbmc-cpp/unordered_map
                      esbmc-cpp/multimap
                      esbmc-cpp/multiset
                      esbmc-cpp/priority_queue
                      esbmc-cpp/queue
                      esbmc-cpp/set
                      esbmc-cpp/unordered_set
                      esbmc-cpp/stack
                      esbmc-cpp/unix
                      esbmc-cpp/vector
                      esbmc-cpp/try_catch
                      esbmc-cpp/functional
                      esbmc-cpp/bitset
                      esbmc-cpp/unwindsetname
                      )
# list of C++11 test suites
set(REGRESSIONS_CPP11 esbmc-cpp11/alignas
                      esbmc-cpp11/array
                      esbmc-cpp11/constructors
                      esbmc-cpp11/cpp
                      esbmc-cpp11/lambda
                      esbmc-cpp11/new-delete
                      esbmc-cpp11/reference
                      esbmc-cpp11/template
                      esbmc-cpp11/tuple
                      )

# list of C++14 test suites
set(REGRESSIONS_CPP14 esbmc-cpp14/cpp
                      esbmc-cpp14/deduced-return-type
                      esbmc-cpp14/lambda
                      esbmc-cpp14/template
                      )

# list of C++17 test suites
set(REGRESSIONS_CPP17 esbmc-cpp17/cpp
                      esbmc-cpp17/lambda
                      )

# list of C++20 test suites
set(REGRESSIONS_CPP20 esbmc-cpp20/cpp)

# enable test case geeneration
set(TEST_GENERATION witnesses/test_case_generation)

# NOTE: In order to make the best of the concurrency set sort the tests from the slowest to fastest.
if(APPLE)
    set(REGRESSIONS esbmc-unix
                    esbmc-unix2
                    esbmc
                    ${REGRESSIONS_SOLIDITY}
                    cbmc
                    cstd
                    llvm
                    floats
                    floats-regression
                    ${REGRESSIONS_JIMPLE}
                    k-induction
                    k-induction-parallel
                    termination
                    nonz3
                    ${REGRESSIONS_BITWUZLA}
                    ${REGRESSIONS_CVC}
                    ${REGRESSIONS_MATHSAT}
                    ${REGRESSIONS_Z3}
                    incremental-smt
                    ${REGRESSIONS_CPP03}
                    ${REGRESSIONS_CPP11}
                    ${REGRESSIONS_CPP14}
                    ${REGRESSIONS_CPP17}
                    ${REGRESSIONS_CPP20}
                    extensions
                    ${REGRESSIONS_CHERI}
                    ${REGRESSIONS_PYTHON}
                    ${REGRESSIONS_PYTHON_INTENSIVE}
                    ${REGRESSIONS_MOPSA}
                    ${REGRESSIONS_NUMPY}
                    ${REGRESSIONS_HUMANEVAL}
                    ${REGRESSIONS_QUIXBUGS}
                    ${REGRESSIONS_PYTHON_COVERAGE}
                    ${REGRESSIONS_AI_GENERATED_CODE}
                    ${TEST_GENERATION}
                    ltl
                    goto-transcoder
                    loop-invariants
       )
elseif(WIN32)
    # FUTURE: Add floats-regression esbmc-cpp/cpp
    set(REGRESSIONS esbmc
                    cbmc
                    cstd
                    llvm
                    floats
                    k-induction
                    termination
                    windows
                    ${REGRESSIONS_JIMPLE}
                    ${REGRESSIONS_BITWUZLA}
                    ${REGRESSIONS_CVC}
                    ${REGRESSIONS_MATHSAT}
                    ${REGRESSIONS_SMTLIB}
                    ${REGRESSIONS_Z3}
                    ltl
                    goto-transcoder
       )
else()
  if(NOT BENCHBRINGUP)
    set(REGRESSIONS esbmc-unix
                    esbmc-unix2
                    esbmc
                    ${REGRESSIONS_SOLIDITY}
                    ${REGRESSIONS_OLD_FRONTEND}
                    cbmc
                    cstd
                    llvm
                    floats
                    floats-regression
                    k-induction
                    termination
                    csmith
                    k-induction-parallel
                    cuda/benchmarks
                    cuda/Supported_long_time
                    nonz3
                    ${REGRESSIONS_BITWUZLA}
                    ${REGRESSIONS_CVC}
                    ${REGRESSIONS_MATHSAT}
                    ${REGRESSIONS_SMTLIB}
                    ${REGRESSIONS_Z3}
                    incremental-smt
                    ${REGRESSIONS_CPP20}
                    ${REGRESSIONS_CPP17}
                    ${REGRESSIONS_CPP14}
                    ${REGRESSIONS_CPP11}
                    ${REGRESSIONS_CPP03}
                    ${REGRESSIONS_GOTO_CONTRACTOR}
                    ${REGRESSIONS_JIMPLE}
                    extensions
                    cuda/COM_sanity_checks
                    linux
                    ${REGRESSIONS_CHERI}
                    ${REGRESSIONS_PYTHON}
                    ${REGRESSIONS_PYTHON_INTENSIVE}
                    ${REGRESSIONS_MOPSA}
                    ${REGRESSIONS_NUMPY}
                    ${REGRESSIONS_HUMANEVAL}
                    ${REGRESSIONS_QUIXBUGS}
                    ${REGRESSIONS_PYTHON_COVERAGE}
                    ${REGRESSIONS_AI_GENERATED_CODE}
                    ${TEST_GENERATION}
                    ltl
                    goto-coverage
                    goto-transcoder
                    parallel-solving
                    loop-invariants
                    function_contract
                    witnesses_validate
       )
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
    if(CORE_REGRESSION_ONLY)
      set(MODES CORE)
    endif()
    add_esbmc_regression("${regression}" "${MODES}")
endforeach()

# Tests that legitimately run close to the per-test TIMEOUT on CI's Linux
# runners (locally ~20-30s; CI ~75-110s; CI cap 120s, leaving no headroom).
# Double their TIMEOUT relative to the global cap so transient CI slowness
# doesn't flap them, while keeping the rest of the suite's fail-fast budget
# unchanged.
math(EXPR ESBMC_REGRESS_TIMEOUT_SLOW "${ESBMC_REGRESS_TIMEOUT} * 2")
set(_slow_regression_tests
    "regression/humaneval/humaneval_34"
    "regression/humaneval/humaneval_54"
    # humaneval_111 / _153 / _161 used to fast-crash with
    # "X() requires constant string" during conversion; with the
    # consteval+nondet fallback in PR #4811 they now reach symex and
    # need the extra budget to finish on slower CI runners.
    "regression/humaneval/humaneval_111"
    "regression/humaneval/humaneval_129"
    "regression/humaneval/humaneval_153"
    "regression/humaneval/humaneval_161"
    "regression/quixbugs/flatten_fail"
    "regression/quixbugs/next_palindrome"
    "regression/quixbugs/pascal"
    "regression/ai-generated-code/two_sum"
    # knapsack(_fail) used to fast-fail with "max() arguments must be of
    # comparable types" because the nested-list element of a defaultdict was
    # mistyped. With the call-site nested-generic inference in #4830 the
    # frontend now resolves the element types and ESBMC reaches the real
    # (defaultdict + nested loop) symex, which does not converge under
    # --incremental-bmc. These remain KNOWNBUG; the larger inner timeout lets
    # testing_tool record the KNOWNBUG verdict instead of ctest hard-killing
    # the run (see shortest_path_lengths note below; #4927).
    "regression/quixbugs/knapsack"
    "regression/quixbugs/knapsack_fail"
    # problem1_fail runs the real DP body (nested loops + dp[-1] + max over a
    # defaultdict-free list). It does not converge under --unwind 9, so it is
    # marked FUTURE (BMC scalability); the larger inner timeout lets
    # testing_tool record the FUTURE verdict instead of ctest hard-killing it.
    "regression/python-intensive/problem1_fail"
    # next_permutation(_fail) used to fast-crash with "Tuple unpacking only
    # supports simple names, not Subscript". With subscript-target unpacking
    # support (#4792) the swap converts and ESBMC reaches the real nested-loop
    # symex (plus reversed()/slice-assign), which does not converge. These stay
    # KNOWNBUG; the larger inner timeout lets testing_tool record the KNOWNBUG
    # verdict instead of ctest hard-killing the run.
    "regression/quixbugs/next_permutation"
    "regression/quixbugs/next_permutation_fail"
)
math(EXPR ESBMC_REGRESS_CTEST_TIMEOUT_SLOW
     "${ESBMC_REGRESS_TIMEOUT_SLOW} + 30")

# Tests that exceed even the 2x slow tier on Debug+Opt Linux runners.
# shortest_path_lengths(_fail) used to fast-fail because defaultdict with
# a lambda factory was unsupported; with #4765 the lambda factory is now
# honoured and ESBMC actually verifies the triple-nested Floyd-Warshall
# loop. Observed CI timings: shortest_path_lengths 244.8s (passed by
# 0.2s under the 245s wall), shortest_path_lengths_fail 245.3s (flapped
# over). Codecov config takes ~302s. Give 3x headroom so transient
# scheduling jitter doesn't tip them over.
math(EXPR ESBMC_REGRESS_TIMEOUT_VERY_SLOW "${ESBMC_REGRESS_TIMEOUT} * 3")
math(EXPR ESBMC_REGRESS_CTEST_TIMEOUT_VERY_SLOW
     "${ESBMC_REGRESS_TIMEOUT_VERY_SLOW} + 30")
set(_very_slow_regression_tests
    "regression/quixbugs/shortest_path_lengths"
    "regression/quixbugs/shortest_path_lengths_fail"
)

foreach(_t IN LISTS _slow_regression_tests)
    if(TEST ${_t})
        # Three coupled budgets to keep aligned:
        #  - ctest TIMEOUT (outer): hard kill -- if it fires first the
        #    test is reported FAILED by ctest, the testing_tool's
        #    KNOWNBUG regex check never runs, and the build fails.
        #  - ESBMC_REGRESS_TIMEOUT env (inner): testing_tool's own
        #    timeout. When it fires, testing_tool gracefully terminates
        #    the ESBMC subprocess and continues to do its regex check
        #    against partial output, so a KNOWNBUG test correctly
        #    decides "no SUCCESSFUL printed -> KNOWNBUG satisfied".
        # The outer must comfortably exceed the inner so testing_tool
        # wins the race (kill + grace + regex check time).
        set_tests_properties(${_t} PROPERTIES
          TIMEOUT ${ESBMC_REGRESS_CTEST_TIMEOUT_SLOW}
          ENVIRONMENT "ESBMC_REGRESS_TIMEOUT=${ESBMC_REGRESS_TIMEOUT_SLOW};ESBMC_REGRESS_MEMORY_LIMIT=${ESBMC_REGRESS_MEMORY_LIMIT}"
        )
    endif()
endforeach()
foreach(_t IN LISTS _very_slow_regression_tests)
    if(TEST ${_t})
        set_tests_properties(${_t} PROPERTIES
          TIMEOUT ${ESBMC_REGRESS_CTEST_TIMEOUT_VERY_SLOW}
          ENVIRONMENT "ESBMC_REGRESS_TIMEOUT=${ESBMC_REGRESS_TIMEOUT_VERY_SLOW};ESBMC_REGRESS_MEMORY_LIMIT=${ESBMC_REGRESS_MEMORY_LIMIT}"
        )
    endif()
endforeach()
