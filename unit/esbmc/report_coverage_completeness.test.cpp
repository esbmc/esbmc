/*******************************************************************
 Module: report_coverage_completeness unit tests
 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <esbmc/bmc.h>
#include <goto-programs/goto_coverage.h>
#include <util/message/message.h>

#include <cstdio>
#include <string>

namespace
{
using claimt = std::pair<std::string, std::string>;

/// Runs a coverage measurement and then its completeness qualifier, returning
/// everything the pair logged.
///
/// The qualifier reads file-static state that only a measurement sets, and
/// that nothing outside multi_property_check clears, so each case runs its own
/// measurement rather than inheriting one. ctest gives every case its own
/// process; run the binary directly and they share one, in declaration order.
std::string measure_then_qualify(size_t total_instances, size_t reached)
{
  const claimt claim{"assertion failed", "t.c:5"};
  goto_coveraget::all_claims = {claim};
  goto_coveraget::total_assert = 1;
  goto_coveraget::total_assert_ins = total_instances;

  optionst options;
  options.set_option("assertion-coverage", true);
  std::unordered_set<std::string> reached_claims;
  std::unordered_multiset<std::string> instances;
  for (size_t i = 0; i < reached; ++i)
    instances.insert(claim.first + "\t" + claim.second);

  pytest_generator pytest_gen;
  ctest_generator ctest_gen;

  FILE *captured = tmpfile();
  REQUIRE(captured != nullptr);
  FILE *previous = messaget::state.out;
  messaget::state.out = captured;
  report_coverage(options, reached_claims, instances, pytest_gen, ctest_gen);
  report_coverage_completeness();
  messaget::state.out = previous;

  std::string logged;
  char buffer[4096];
  rewind(captured);
  for (size_t n; (n = fread(buffer, 1, sizeof(buffer), captured)) > 0;)
    logged.append(buffer, n);
  fclose(captured);
  return logged;
}
/// Everything report_coverage_completeness logs on its own.
std::string qualify_only()
{
  FILE *captured = tmpfile();
  REQUIRE(captured != nullptr);
  FILE *previous = messaget::state.out;
  messaget::state.out = captured;
  report_coverage_completeness();
  messaget::state.out = previous;

  std::string logged;
  char buffer[1024];
  rewind(captured);
  for (size_t n; (n = fread(buffer, 1, sizeof(buffer), captured)) > 0;)
    logged.append(buffer, n);
  fclose(captured);
  return logged;
}
} // namespace

// Declared first: nothing measured is the state the process starts in, and no
// measurement this binary runs is ever undone.
TEST_CASE("nothing measured qualifies nothing", "[coverage]")
{
  CHECK(qualify_only().empty());
}

TEST_CASE("a measurement that added up is complete", "[coverage]")
{
  CHECK_THAT(
    measure_then_qualify(2, 1),
    Catch::Matchers::Contains("COVERAGE ANALYSIS COMPLETE"));
}

TEST_CASE("an unknown instance total makes it incomplete", "[coverage]")
{
  // The percentages are lower bounds once a total could not be determined,
  // and the reason is named so the reader can act on it.
  const std::string logged = measure_then_qualify(1, 2);
  CHECK_THAT(
    logged,
    Catch::Matchers::Contains(
      "COVERAGE ANALYSIS INCOMPLETE: the percentages above are lower bounds"));
  CHECK_THAT(logged, Catch::Matchers::Contains("reason:"));
  CHECK_THAT(
    logged,
    Catch::Matchers::Contains(
      "the total number of assertion instances could not be determined"));
}
