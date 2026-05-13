new_unit_test(irep2test "irep2.test.cpp" "util_esbmc")

# Microbenchmarks: built but not discovered as ctest cases. Run locally
# with: ./unit/irep2/irep2bench '[bench]' to capture before/after numbers
# for Track D perf work.
add_executable(irep2bench irep2.bench.cpp)
target_include_directories(irep2bench PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(irep2bench
  PRIVATE util_esbmc Catch2::Catch2 ${OS_INCLUDE_LIBS})
