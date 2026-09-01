/*******************************************************************
 Module: bmct::report_result unit tests
 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <esbmc/bmc.h>
#include <solvers/smt/smt_result.h>
#include <util/message/message.h>

#include <cstdio>
#include <initializer_list>
#include <string>

namespace
{
struct reportert : bmct
{
  reportert(goto_functionst &funcs, optionst &opts, contextt &context)
    : bmct(funcs, opts, context)
  {
  }
  using bmct::report_result;
};

optionst flags(std::initializer_list<const char *> set)
{
  optionst options;
  // Keeps the constructor off the on-disk SSA cache.
  options.set_option("no-cache-asserts", true);
  for (const char *flag : set)
    options.set_option(flag, true);
  return options;
}

/// Everything report_result logs for this combination, captured from the
/// message system's own output stream.
std::string verdict(optionst &options, smt_resultt result)
{
  goto_functionst goto_functions;
  contextt context;
  reportert bmc(goto_functions, options, context);

  FILE *captured = tmpfile();
  REQUIRE(captured != nullptr);
  FILE *previous = messaget::state.out;
  messaget::state.out = captured;
  bmc.report_result(result);
  messaget::state.out = previous;

  std::string logged;
  char buffer[4096];
  rewind(captured);
  for (size_t n; (n = fread(buffer, 1, sizeof(buffer), captured)) > 0;)
    logged.append(buffer, n);
  fclose(captured);
  return logged;
}

std::string verdict(optionst &&options, smt_resultt result)
{
  return verdict(options, result);
}
} // namespace

TEST_CASE("an unsatisfiable run reports success", "[bmc][report]")
{
  CHECK_THAT(
    verdict(flags({}), P_UNSATISFIABLE),
    Catch::Matchers::Contains("VERIFICATION SUCCESSFUL"));
}

TEST_CASE("a satisfiable run reports failure", "[bmc][report]")
{
  CHECK_THAT(
    verdict(flags({}), P_SATISFIABLE),
    Catch::Matchers::Contains("VERIFICATION FAILED"));
}

TEST_CASE("a solver error is not a verdict", "[bmc][report]")
{
  const std::string logged = verdict(flags({}), P_ERROR);
  CHECK_THAT(logged, Catch::Matchers::Contains("SMT solver failed"));
  CHECK_THAT(logged, !Catch::Matchers::Contains("VERIFICATION"));
}

TEST_CASE("emitting a formula decides nothing", "[bmc][report]")
{
  CHECK(verdict(flags({}), P_SMTLIB).empty());
}

TEST_CASE("runs that report their own verdict stay quiet", "[bmc][report]")
{
  CHECK(verdict(flags({"k-induction-parallel"}), P_SATISFIABLE).empty());
  CHECK(verdict(flags({"diagnose-unknown-properties"}), P_SATISFIABLE).empty());
  CHECK(verdict(flags({"coverage-measurement"}), P_SATISFIABLE).empty());
}

TEST_CASE("a completed dead-code analysis is a successful run", "[bmc][report]")
{
  // Its probes are violated for every live branch, which must not become a
  // FAILED verdict.
  CHECK_THAT(
    verdict(flags({"dead-code-check"}), P_SATISFIABLE),
    Catch::Matchers::Contains("VERIFICATION SUCCESSFUL"));
  CHECK_THAT(
    verdict(flags({"dead-code-check"}), P_ERROR),
    Catch::Matchers::Contains("SMT solver failed"));
  CHECK(verdict(flags({"dead-code-check"}), P_SMTLIB).empty());
}

TEST_CASE("a proved termination property is a failure", "[bmc][report]")
{
  CHECK_THAT(
    verdict(flags({"inductive-step", "termination"}), P_UNSATISFIABLE),
    Catch::Matchers::Contains("VERIFICATION FAILED"));
}

TEST_CASE("a clean base case proves nothing on its own", "[bmc][report]")
{
  const std::string logged = verdict(flags({"base-case"}), P_UNSATISFIABLE);
  CHECK_THAT(
    logged,
    Catch::Matchers::Contains("No bug has been found in the base case"));
  CHECK_THAT(logged, !Catch::Matchers::Contains("VERIFICATION SUCCESSFUL"));
}

TEST_CASE("multi-property reports success from the base case", "[bmc][report]")
{
  CHECK_THAT(
    verdict(flags({"base-case", "multi-property"}), P_UNSATISFIABLE),
    Catch::Matchers::Contains("VERIFICATION SUCCESSFUL"));
}

TEST_CASE("an earlier k-step violation suppresses success", "[bmc][report]")
{
  CHECK(verdict(flags({"kind-violation-found"}), P_UNSATISFIABLE).empty());
}

TEST_CASE("an incomplete inductive step proves nothing", "[bmc][report]")
{
  CHECK(verdict(
          flags({"inductive-step", "disable-inductive-step"}), P_UNSATISFIABLE)
          .empty());
}

TEST_CASE("a bounded round withholds the verdict", "[bmc][report]")
{
  const std::string logged =
    verdict(flags({"suppress-bounded-success"}), P_UNSATISFIABLE);
  CHECK_THAT(
    logged,
    Catch::Matchers::Contains(
      "No violation found within the current context bound"));
  CHECK_THAT(logged, !Catch::Matchers::Contains("VERIFICATION SUCCESSFUL"));
}

TEST_CASE("neither half of k-induction claims a violation", "[bmc][report]")
{
  CHECK_THAT(
    verdict(flags({"forward-condition"}), P_SATISFIABLE),
    Catch::Matchers::Contains(
      "The forward condition is unable to prove the property"));
  CHECK_THAT(
    verdict(flags({"inductive-step"}), P_SATISFIABLE),
    Catch::Matchers::Contains(
      "The inductive step is unable to prove the property"));
}
