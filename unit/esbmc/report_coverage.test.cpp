/*******************************************************************
 Module: report_coverage unit tests
 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <esbmc/bmc.h>
#include <goto-programs/goto_coverage.h>
#include <util/message/message.h>

#include <cstdio>
#include <initializer_list>
#include <set>
#include <string>

#include <unordered_set>

namespace
{
using claimt = std::pair<std::string, std::string>;

std::string signature(const claimt &claim)
{
  return claim.first + "\t" + claim.second;
}

/// The counters report_coverage reads are process-wide, so every case states
/// all of them rather than inheriting whatever ran before it.
void reset_totals()
{
  goto_coveraget::total_assert = 0;
  goto_coveraget::total_assert_ins = 0;
  goto_coveraget::total_branch = 0;
  goto_coveraget::total_func_branch = 0;
  goto_coveraget::total_kpath = 0;
  goto_coveraget::total_kpath_spanning = 0;
  goto_coveraget::total_cond.clear();
  goto_coveraget::all_claims.clear();
  goto_coveraget::k_path_spanning_redundant.clear();
}

optionst flags(std::initializer_list<const char *> set)
{
  optionst options;
  for (const char *flag : set)
    options.set_option(flag, true);
  return options;
}

/// Everything report_coverage logs, captured from the message system's own
/// output stream.
std::string coverage_report(
  const optionst &options,
  std::unordered_set<std::string> reached,
  const std::unordered_multiset<std::string> &reached_instances = {})
{
  pytest_generator pytest_gen;
  ctest_generator ctest_gen;

  FILE *captured = tmpfile();
  REQUIRE(captured != nullptr);
  FILE *previous = messaget::state.out;
  messaget::state.out = captured;
  report_coverage(options, reached, reached_instances, pytest_gen, ctest_gen);
  messaget::state.out = previous;

  std::string logged;
  char buffer[4096];
  rewind(captured);
  for (size_t n; (n = fread(buffer, 1, sizeof(buffer), captured)) > 0;)
    logged.append(buffer, n);
  fclose(captured);
  return logged;
}
} // namespace

TEST_CASE("dead-code analysis reports no coverage block", "[coverage]")
{
  // It borrows the coverage instrumentation but reports CWE-561 advisories
  // from start_bmc instead, once exploration has finished. Coverage is asked
  // for as well, so silence here is the guard rather than an idle run.
  const claimt claim{"assertion failed", "t.c:5"};
  reset_totals();
  goto_coveraget::all_claims = {claim};
  goto_coveraget::total_assert = 1;
  goto_coveraget::total_assert_ins = 1;

  CHECK(coverage_report(flags({"dead-code-check", "assertion-coverage"}), {})
          .empty());
}

TEST_CASE("assertion coverage counts claims and instances", "[coverage]")
{
  const claimt reached{"assertion failed", "t.c:5"};
  const claimt missed{"division by zero", "t.c:9"};
  reset_totals();
  goto_coveraget::all_claims = {reached, missed};
  goto_coveraget::total_assert = 2;
  goto_coveraget::total_assert_ins = 4;

  const std::string logged =
    coverage_report(flags({"assertion-coverage"}), {}, {signature(reached)});

  CHECK_THAT(logged, Catch::Matchers::Contains("[Coverage]"));
  CHECK_THAT(logged, Catch::Matchers::Contains("Total Asserts: 2"));
  CHECK_THAT(logged, Catch::Matchers::Contains("Unreached Asserts: 1"));
  CHECK_THAT(logged, Catch::Matchers::Contains("Total Assertion Instances: 4"));
  CHECK_THAT(
    logged, Catch::Matchers::Contains("Reached Assertion Instances: 1"));
  CHECK_THAT(
    logged, Catch::Matchers::Contains("Assertion Instances Coverage: 25"));
}

TEST_CASE("assertion coverage can list every claim", "[coverage]")
{
  const claimt reached{"assertion failed", "t.c:5"};
  const claimt missed{"division by zero", "t.c:9"};
  reset_totals();
  goto_coveraget::all_claims = {reached, missed};
  goto_coveraget::total_assert = 2;
  goto_coveraget::total_assert_ins = 2;

  const std::string logged = coverage_report(
    flags({"assertion-coverage-claims"}), {}, {signature(reached)});

  CHECK_THAT(logged, Catch::Matchers::Contains("assertion failed"));
  CHECK_THAT(logged, Catch::Matchers::Contains(": REACHED"));
  CHECK_THAT(logged, Catch::Matchers::Contains("division by zero"));
  CHECK_THAT(logged, Catch::Matchers::Contains(": UNREACHED"));
}

TEST_CASE(
  "more instances reached than counted is not a percentage",
  "[coverage]")
{
  // A loop too large or too non-deterministic to goto-unwind leaves the
  // instance total below what symex actually reached.
  const claimt first{"assertion failed", "t.c:5"};
  const claimt second{"assertion failed", "t.c:6"};
  reset_totals();
  goto_coveraget::all_claims = {first, second};
  goto_coveraget::total_assert = 2;
  goto_coveraget::total_assert_ins = 1;

  const std::string logged = coverage_report(
    flags({"assertion-coverage"}), {}, {signature(first), signature(second)});

  CHECK_THAT(
    logged,
    Catch::Matchers::Contains(
      "Total Assertion Instances: unknown / non-deterministic"));
  CHECK_THAT(
    logged, Catch::Matchers::Contains("Assertion Instances Coverage Unknown"));
}

TEST_CASE("a program with no assertion instances covers none", "[coverage]")
{
  reset_totals();
  goto_coveraget::total_assert = 0;
  goto_coveraget::total_assert_ins = 0;

  CHECK_THAT(
    coverage_report(flags({"assertion-coverage"}), {}),
    Catch::Matchers::Contains("Assertion Instances Coverage: 0%"));
}

TEST_CASE("an unreached negation is an unsatisfied condition", "[coverage]")
{
  // Each condition is instrumented as a pair: the claim and its negation.
  // Reaching only one half satisfies one and leaves the other unsatisfied.
  const claimt condition{"a == 1", "t.c:3"};
  reset_totals();
  goto_coveraget::total_cond = {condition};

  const std::string logged =
    coverage_report(flags({"condition-coverage"}), {signature(condition)});

  CHECK_THAT(logged, Catch::Matchers::Contains("Reached Conditions:  2"));
  CHECK_THAT(
    logged, Catch::Matchers::Contains("Condition Properties - SATISFIED:  1"));
  CHECK_THAT(
    logged,
    Catch::Matchers::Contains("Condition Properties - UNSATISFIED:  1"));
  CHECK_THAT(logged, Catch::Matchers::Contains("Condition Coverage: 100%"));
}

TEST_CASE("a condition never reached was short circuited", "[coverage]")
{
  const claimt condition{"a == 1", "t.c:3"};
  reset_totals();
  goto_coveraget::total_cond = {condition};

  const std::string logged = coverage_report(flags({"condition-coverage"}), {});

  CHECK_THAT(logged, Catch::Matchers::Contains("Reached Conditions:  0"));
  CHECK_THAT(
    logged, Catch::Matchers::Contains("Short Circuited Conditions:  1"));
  CHECK_THAT(logged, Catch::Matchers::Contains("Condition Coverage: 0%"));
}

TEST_CASE("a program with no conditions covers none", "[coverage]")
{
  reset_totals();
  CHECK_THAT(
    coverage_report(flags({"condition-coverage"}), {}),
    Catch::Matchers::Contains("Condition Coverage: 0%"));
}

TEST_CASE("branch coverage counts the goals reached", "[coverage]")
{
  const claimt taken{"branch taken", "t.c:4"};
  const claimt missed{"branch not taken", "t.c:4"};
  reset_totals();
  goto_coveraget::all_claims = {taken, missed};
  goto_coveraget::total_branch = 4;

  const std::string logged =
    coverage_report(flags({"branch-coverage"}), {signature(taken)});

  CHECK_THAT(logged, Catch::Matchers::Contains("Branches : 4"));
  CHECK_THAT(logged, Catch::Matchers::Contains("Reached : 1"));
  CHECK_THAT(logged, Catch::Matchers::Contains("Branch Coverage: 25%"));
}

TEST_CASE("branch coverage can list what was reached", "[coverage]")
{
  const claimt taken{"branch taken", "t.c:4"};
  reset_totals();
  goto_coveraget::all_claims = {taken};
  goto_coveraget::total_branch = 1;

  CHECK_THAT(
    coverage_report(flags({"branch-coverage-claims"}), {signature(taken)}),
    Catch::Matchers::Contains("branch taken"));
}

TEST_CASE("a program with no branches has no percentage", "[coverage]")
{
  reset_totals();
  goto_coveraget::total_func_branch = 0;

  CHECK_THAT(
    coverage_report(flags({"branch-function-coverage"}), {}),
    Catch::Matchers::Contains("Branch Coverage: N/A (no branches)"));
}

TEST_CASE("function branch coverage counts entry points too", "[coverage]")
{
  const claimt entry{"function main entered", "t.c:1"};
  reset_totals();
  goto_coveraget::all_claims = {entry};
  goto_coveraget::total_func_branch = 2;

  const std::string logged =
    coverage_report(flags({"branch-function-coverage"}), {signature(entry)});

  CHECK_THAT(
    logged, Catch::Matchers::Contains("Function Entry Points & Branches : 2"));
  CHECK_THAT(logged, Catch::Matchers::Contains("Branch Coverage: 50%"));
}

TEST_CASE("k-path coverage measures against the spanning set", "[coverage]")
{
  const claimt maximal{"k-path witness", "t.c:7"};
  reset_totals();
  goto_coveraget::total_kpath = 5;
  goto_coveraget::total_kpath_spanning = 2;

  const std::string logged =
    coverage_report(flags({"k-path-coverage-enabled"}), {signature(maximal)});

  CHECK_THAT(logged, Catch::Matchers::Contains("k-Path Witnesses : 5"));
  CHECK_THAT(logged, Catch::Matchers::Contains("Spanning Set : 2"));
  CHECK_THAT(logged, Catch::Matchers::Contains("Reached : 1"));
  CHECK_THAT(logged, Catch::Matchers::Contains("k-Path Coverage: 50%"));
}

TEST_CASE("a subsumed k-path goal does not inflate coverage", "[coverage]")
{
  // Numerator and denominator must both restrict to maximal goals, or a
  // reached-but-subsumed goal counts against a maximal-only total.
  const claimt maximal{"k-path witness", "t.c:7"};
  const claimt subsumed{"k-path witness", "t.c:8"};
  reset_totals();
  goto_coveraget::total_kpath = 5;
  goto_coveraget::total_kpath_spanning = 2;
  goto_coveraget::k_path_spanning_redundant = {subsumed};

  CHECK_THAT(
    coverage_report(
      flags({"k-path-coverage-enabled"}),
      {signature(maximal), signature(subsumed)}),
    Catch::Matchers::Contains("Reached : 1"));
}

TEST_CASE("no k-path goals means no percentage", "[coverage]")
{
  reset_totals();
  goto_coveraget::total_kpath = 3;
  goto_coveraget::total_kpath_spanning = 0;

  CHECK_THAT(
    coverage_report(flags({"k-path-coverage-enabled"}), {}),
    Catch::Matchers::Contains("k-Path Coverage: N/A (no k-path goals)"));
}
