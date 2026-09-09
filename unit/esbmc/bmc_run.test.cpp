/*******************************************************************
 Module: bmct end-to-end run unit tests

 One in-process verification of a few lines of C: parse, goto convert,
 symex, encode, solve and report. Milliseconds, and the only way to reach
 the run/solve path -- everything below multi_property_check needs a live
 reachability tree and a real solver.
 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <esbmc/bmc.h>
#include <solvers/smt/smt_result.h>
#include <util/message/message.h>

#include <cstdio>
#include <initializer_list>
#include <string>

#include "../testing-utils/goto_factory.h"

namespace
{
struct runt
{
  smt_resultt result;
  std::string output;
};

/// Verifies \p src and returns the verdict together with everything the run
/// logged, captured from the message system's own output stream.
runt verify(
  const std::string &src,
  std::initializer_list<const char *> set = {})
{
  std::string source = src;
  program prog = goto_factory::get_goto_functions(
    source, goto_factory::Architecture::BIT_64);
  optionst options =
    goto_factory::get_default_options(goto_factory::get_default_cmdline("t.c"));
  for (const char *flag : set)
    options.set_option(flag, true);

  bmct bmc(prog.functions, options, prog.context);

  FILE *captured = tmpfile();
  REQUIRE(captured != nullptr);
  FILE *previous = messaget::state.out;
  messaget::state.out = captured;
  const smt_resultt result = bmc.start_bmc();
  messaget::state.out = previous;

  std::string logged;
  char buffer[8192];
  rewind(captured);
  for (size_t n; (n = fread(buffer, 1, sizeof(buffer), captured)) > 0;)
    logged.append(buffer, n);
  fclose(captured);
  return {result, logged};
}
} // namespace

TEST_CASE("an assertion that holds verifies", "[bmc][run]")
{
  const runt run = verify("int main() { int x = 1; assert(x == 1); }");
  CHECK(run.result == P_UNSATISFIABLE);
  CHECK_THAT(run.output, Catch::Matchers::Contains("VERIFICATION SUCCESSFUL"));
  CHECK_THAT(run.output, Catch::Matchers::Contains("PASSED"));
}

TEST_CASE("an assertion that fails is refuted", "[bmc][run]")
{
  const runt run = verify("int main() { int x = 1; assert(x == 2); }");
  CHECK(run.result == P_SATISFIABLE);
  CHECK_THAT(run.output, Catch::Matchers::Contains("VERIFICATION FAILED"));
  CHECK_THAT(run.output, Catch::Matchers::Contains("[Counterexample]"));
}

TEST_CASE("a nondet input is refuted with a witness", "[bmc][run]")
{
  const runt run = verify(
    "int nondet_int();\n"
    "int main() { int x = nondet_int(); assert(x != 42); }");
  CHECK(run.result == P_SATISFIABLE);
  CHECK_THAT(run.output, Catch::Matchers::Contains("VERIFICATION FAILED"));
}

TEST_CASE("a built-in check is proved alongside the assertion", "[bmc][run]")
{
  // The property report names the division check, not just the assertion.
  const runt run = verify(
    "int nondet_int();\n"
    "int main() { int d = nondet_int(); if (d != 0) { int q = 7 / d; "
    "assert(q * d <= 7); } }");
  CHECK_THAT(run.output, Catch::Matchers::Contains("division by zero"));
}

TEST_CASE("every property gets its own verdict", "[bmc][run]")
{
  const runt run = verify(
    "int nondet_int();\n"
    "int main() { int x = nondet_int(); assert(x == x); assert(x != x); }",
    {"multi-property"});
  CHECK_THAT(run.output, Catch::Matchers::Contains("PASSED"));
  CHECK_THAT(run.output, Catch::Matchers::Contains("FAILED"));
}

TEST_CASE("a coverage run measures instead of verifying", "[bmc][run]")
{
  // The instrumentation replaced the program's assertions with reachability
  // probes, so the run reports goals rather than a verdict.
  const runt run = verify(
    "int main() { int x = 1; assert(x == 1); }",
    {"assertion-coverage", "coverage-measurement"});
  CHECK_THAT(run.output, Catch::Matchers::Contains("Coverage goals:"));
  CHECK_THAT(run.output, !Catch::Matchers::Contains("VERIFICATION SUCCESSFUL"));
}
