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
# Grace the ctest TIMEOUT property gets *on top of* the runner's own
# ESBMC_REGRESS_TIMEOUT budget. The runner (testing_tool.py) enforces the real
# budget internally: it kills ESBMC at ESBMC_REGRESS_TIMEOUT and, for
# KNOWNBUG/FUTURE tests, accepts the timeout as satisfying the expected-fail
# contract instead of erroring. That accept-and-return path needs a few seconds
# (SIGTERM grace + final communicate) to complete. If ctest's own TIMEOUT
# property equals ESBMC_REGRESS_TIMEOUT, ctest kills the runner mid-teardown and
# reports a hard "Timeout" before the accept logic can run, defeating it. The
# grace keeps ESBMC's compute budget unchanged while letting the runner finish.
set(ESBMC_REGRESS_TIMEOUT_GRACE 30 CACHE STRING
    "Seconds added to the ctest TIMEOUT property over the runner's own budget")
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

# Computed after any BENCHBRINGUP override so it tracks the final budget.
math(EXPR ESBMC_REGRESS_CTEST_TIMEOUT
     "${ESBMC_REGRESS_TIMEOUT} + ${ESBMC_REGRESS_TIMEOUT_GRACE}")

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
      TIMEOUT ${ESBMC_REGRESS_CTEST_TIMEOUT}
      ENVIRONMENT "ESBMC_REGRESS_TIMEOUT=${ESBMC_REGRESS_TIMEOUT};ESBMC_REGRESS_MEMORY_LIMIT=${ESBMC_REGRESS_MEMORY_LIMIT}"
      LABELS "regression;${folder}/")
endfunction()

function(add_esbmc_regression folder modes)
    subdirlist(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR}/${regression})
    if(${folder} STREQUAL "quixbugs")
        list(REMOVE_ITEM SUBDIRS bucketsort)
    endif()
    if(${folder} STREQUAL "ld")
        # WP3 benchmark dataset, not a CTest test (no test.desc)
        list(REMOVE_ITEM SUBDIRS benchmarks)
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
if(ENABLE_LD_FRONTEND)
    set(REGRESSIONS_LD ld)
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
                    ${REGRESSIONS_LD}
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
                    ${REGRESSIONS_LD}
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
    # humaneval_109 (move_one_ball) verifies SUCCESSFUL but is slow: ~32s on a
    # fast local build, and it exceeds the 120s per-test cap on the DebugOpt
    # static Linux runner (sorted()/min()/index()/slice+concat over a list under
    # --smt-during-symex --smt-symex-guard --bitwuzla). Give it the 2x budget so
    # transient CI slowness doesn't time it out (#5189).
    "regression/humaneval/humaneval_109"
    # humaneval_90 used to fast-crash with the member2t assertion
    # ("source->type->type_id == struct_id ...") when next_smallest()'s
    # inline optional comparison (None | int == 2) was lowered. With the
    # materialise-before-member fix the crash is gone and ESBMC reaches the
    # real sorted(set(lst)) symex, which does not converge under the default
    # budget. It stays KNOWNBUG; the larger inner timeout lets testing_tool
    # record the KNOWNBUG verdict instead of ctest hard-killing the run.
    "regression/humaneval/humaneval_90"
    # humaneval_158 used to fast-abort with "__ESBMC_list_size ... got empty,
    # expected pointer" when find_max()'s inline sorted(words)[0] subscript was
    # lowered. With the materialise-the-call fix the abort is gone and ESBMC
    # reaches the real sorted-with-key symex, which does not converge under the
    # default budget. It stays KNOWNBUG; the larger inner timeout lets
    # testing_tool record the KNOWNBUG verdict instead of ctest hard-killing.
    "regression/humaneval/humaneval_158"
    # humaneval_39 used to fast-abort in get_significand_width when prime_fib's
    # math.sqrt() was applied to an any-typed value. With the nondet
    # over-approximation fix the abort is gone and ESBMC reaches the real
    # prime_fib symex (a `while True` loop over a growing Fibonacci list),
    # which does not converge under the default budget. It stays KNOWNBUG; the
    # larger inner timeout lets testing_tool record the KNOWNBUG verdict
    # instead of ctest hard-killing the run.
    "regression/humaneval/humaneval_39"
    # humaneval_145 used to fast-fail with "TypeError: str() expects a string
    # argument" when order_by_points()'s nested digits_sum(n) ran `str(n)` on an
    # unresolved (any_type) parameter. With the str()-of-unresolved-arg fix
    # (PR #5225) the spurious error is gone and ESBMC reaches the real
    # order_by_points symex (sorted(key=digits_sum) with a str(int)+int(digit)
    # comprehension under --unwind 20), which does not converge under the
    # default budget. It stays KNOWNBUG; the larger inner timeout lets
    # testing_tool record the KNOWNBUG verdict instead of ctest hard-killing.
    "regression/humaneval/humaneval_145"
    # humaneval_93 used to fast-fail with "Unsupported reassignment from dict to
    # list" when encode()'s `dict([(i, chr(ord(i) + 2)) for i in vowels])` was
    # rejected during conversion. With the dict()-over-comprehension lowering
    # (PR #5223) the spurious error is gone and ESBMC reaches the real encode
    # symex (swapcase + dict-populate loop + join over the message under
    # --unwind 20), which does not converge under the default budget. It stays
    # KNOWNBUG; the larger inner timeout lets testing_tool record the KNOWNBUG
    # verdict instead of ctest hard-killing the run.
    "regression/humaneval/humaneval_93"
    "regression/quixbugs/flatten_fail"
    "regression/quixbugs/next_palindrome"
    "regression/quixbugs/pascal"
    "regression/ai-generated-code/two_sum"
    # lis_fail runs the real longest-increasing-subsequence body (dict +
    # list-comprehension over range + max()) under its
    # --incremental-bmc/--smt-during-symex/--bitwuzla config, landing at ~107s
    # on the DebugOpt Linux runner (~125s locally) -- right at the 120s cap with
    # no headroom -- so transient CI scheduling jitter tips it over (it timed
    # out at 120.02s in #5236, a run that does not touch its code path). Give it
    # the 2x budget like the other borderline quixbugs tests.
    "regression/quixbugs/lis_fail"
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
    # breadth_first_search(_fail) used to fast-fail with
    # "NameError: name 'Queue' is not defined" because aliased deque imports
    # were not resolved. With PR #5008 the alias fix and popleft/appendleft
    # support land, so ESBMC now reaches the real BFS symex (deque + set +
    # 6-node graph), which does not converge under --incremental-bmc due to
    # the __memcpy_impl loop in the list OM (string.c:278). These stay
    # KNOWNBUG; the larger inner timeout lets testing_tool record the KNOWNBUG
    # verdict instead of ctest hard-killing the run.
    "regression/quixbugs/breadth_first_search"
    "regression/quixbugs/breadth_first_search_fail"
    # reverse_linked_list used to fast-crash with a SIGSEGV when comparing a
    # None-handle (pointer-width int) against a by-value Node struct via ==.
    # With the pointer-identity reconciliation fix (PR #5274) the crash is gone
    # and ESBMC reaches the real linked-list reversal symex (list-equality +
    # loop scalability wall), which does not converge under --incremental-bmc.
    # It stays KNOWNBUG; the larger inner timeout lets testing_tool record the
    # KNOWNBUG verdict instead of ctest hard-killing the run.
    "regression/quixbugs/reverse_linked_list"
    # github_4435 was a soundness false alarm on the SV-COMP libvsync
    # bounded_mpmc_check_full benchmark; the underlying deref unit
    # mismatch is fixed by PR #4939, but the full benchmark
    # (6055-line preprocessed libvsync) does not converge under
    # `--unlimited-k-steps` on CI runners. Marked FUTURE (BMC scalability);
    # the larger inner timeout lets testing_tool record the FUTURE
    # verdict instead of ctest hard-killing the run.
    "regression/esbmc-unix/github_4435"
)
math(EXPR ESBMC_REGRESS_CTEST_TIMEOUT_SLOW
     "${ESBMC_REGRESS_TIMEOUT_SLOW} + 30")

# Tests that exceed even the 2x slow tier on Debug+Opt Linux runners.
# shortest_path_lengths(_fail) used to fast-fail because defaultdict with
# a lambda factory was unsupported; with #4765 the lambda factory is now
# honoured and ESBMC actually verifies the triple-nested Floyd-Warshall
# loop. shortest_path_lengths has long been KNOWNBUG (non-converging);
# shortest_path_lengths_fail is now KNOWNBUG too because #5236 made the
# `{(i, i): 0 for i in range(n)}` seed populate concrete keys instead of
# nondet ones, so ESBMC reaches the real defaultdict + triple-nested-loop
# symex, which no longer converges under --incremental-bmc (>900s locally,
# was ~245s when the keys were nondet). The 3x inner timeout fires well
# before the program completes and, being shorter than the outer ctest
# TIMEOUT, lets testing_tool record the KNOWNBUG verdict instead of ctest
# hard-killing the run.
math(EXPR ESBMC_REGRESS_TIMEOUT_VERY_SLOW "${ESBMC_REGRESS_TIMEOUT} * 3")
math(EXPR ESBMC_REGRESS_CTEST_TIMEOUT_VERY_SLOW
     "${ESBMC_REGRESS_TIMEOUT_VERY_SLOW} + 30")
set(_very_slow_regression_tests
    "regression/quixbugs/shortest_path_lengths"
    "regression/quixbugs/shortest_path_lengths_fail"
    # humaneval_62 (derivative) verifies SUCCESSFUL but is heavy: its body is a
    # list comprehension over enumerate(xs) followed by a slice
    # ([(i * x) for i, x in enumerate(xs)][1:]) under --unwind 20, which unwinds
    # the list-OM loops 20x and feeds a large bit-vector query (encode ~16s +
    # Bitwuzla ~54s locally). It times out at the 120s per-test cap on the
    # DebugOpt static Linux runner; an assertions-on build measures ~248s, the
    # same band as shortest_path_lengths above. Give it the 3x budget so it
    # doesn't flap.
    "regression/humaneval/humaneval_62"
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
