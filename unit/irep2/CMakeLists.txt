new_unit_test(irep2test "irep2.test.cpp" "util_esbmc")

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
