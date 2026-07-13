new_unit_test(irep2test "irep2.test.cpp" "util_esbmc")
new_unit_test(irep2refcounttest "refcount.test.cpp" "util_esbmc")

# libFuzzer harness driving the real irep_container refcount lifecycle with
# nondeterministic input (built only with -DENABLE_FUZZER=On). The same
# run_ops() driver is replayed deterministically by irep2refcounttest so the
# property is covered in normal CI regardless of the fuzzer.
new_fuzz_test(irep2refcountfuzz "refcount.fuzz.cpp" "util_esbmc")

# Microbenchmarks: built but not discovered as ctest cases. Run locally
# with: ./unit/irep2/irep2bench '[bench]' to capture before/after numbers
# for Track D perf work.
add_executable(irep2bench irep2.bench.cpp)
target_include_directories(irep2bench PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(irep2bench
  PRIVATE util_esbmc Catch2::Catch2 ${OS_INCLUDE_LIBS})

# Guard-specific microbenchmarks. Run locally with:
#   ./unit/irep2/guardbench '[bench]'
add_executable(guardbench guard.bench.cpp)
target_include_directories(guardbench PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(guardbench
  PRIVATE util_esbmc Catch2::Catch2 ${OS_INCLUDE_LIBS})

# Memory-allocation microbenchmarks. Combines per-op latency
# (Catch2 BENCHMARK), allocation count via global new/delete
# override, and VmRSS/VmPeak readings from /proc/self/status. Run
# with:
#   ./unit/irep2/memorybench '[bench]'   # latency
#   ./unit/irep2/memorybench '[probe]'   # alloc counts + RSS
add_executable(memorybench memory.bench.cpp)
target_include_directories(memorybench PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(memorybench
  PRIVATE util_esbmc Catch2::Catch2 ${OS_INCLUDE_LIBS})
