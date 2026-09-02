/*******************************************************************
 Module: bmct::report_multi_property_trace unit tests
 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <esbmc/bmc.h>
#include <irep2/irep2_utils.h>
#include <langapi/mode.h>
#include <solvers/smt/smt_result.h>
#include <util/lang/c_types.h>
#include <util/message/message.h>

#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

// Rendering a witness input goes through from_expr, which falls back to C for
// an unqualified identifier and dereferences whatever new_language() returns.
const mode_table_et mode_table[] = {LANGAPI_MODE_CLANG_C, LANGAPI_MODE_END};

namespace
{
struct reportert : bmct
{
  reportert(goto_functionst &funcs, optionst &opts, contextt &context)
    : bmct(funcs, opts, context)
  {
  }
  using bmct::enumeration_stop_reasont;
  using bmct::report_multi_property_trace;
  using bmct::witness_recordt;
};

using stop_reasont = reportert::enumeration_stop_reasont;

optionst flags(std::initializer_list<const char *> set)
{
  optionst options;
  options.set_option("no-cache-asserts", true);
  for (const char *flag : set)
    options.set_option(flag, true);
  return options;
}

/// A witness carrying \p inputs concrete integer inputs and an empty trace.
reportert::witness_recordt witness(std::initializer_list<int> inputs)
{
  reportert::witness_recordt record;
  record.ce_index = 0;
  for (int value : inputs)
  {
    collected_nondet_value nondet;
    nondet.symbol_name = "nondet$symex::nondet0";
    nondet.type = int_type2();
    nondet.value_expr = constant_int2tc(int_type2(), BigInt(value));
    record.nondet_inputs.push_back(nondet);
  }
  return record;
}

/// Everything report_multi_property_trace logs, captured from the message
/// system's own output stream.
std::string report(
  optionst &options,
  smt_resultt result,
  const std::vector<reportert::witness_recordt> &witnesses,
  stop_reasont stop_reason,
  bool reachability_trace = false)
{
  goto_functionst goto_functions;
  contextt context;
  reportert bmc(goto_functions, options, context);

  FILE *captured = tmpfile();
  REQUIRE(captured != nullptr);
  FILE *previous = messaget::state.out;
  messaget::state.out = captured;
  bmc.report_multi_property_trace(
    result, witnesses, stop_reason, "claim.assertion.1", reachability_trace);
  messaget::state.out = previous;

  std::string logged;
  char buffer[8192];
  rewind(captured);
  for (size_t n; (n = fread(buffer, 1, sizeof(buffer), captured)) > 0;)
    logged.append(buffer, n);
  fclose(captured);
  return logged;
}

std::string report(
  optionst &&options,
  smt_resultt result,
  const std::vector<reportert::witness_recordt> &witnesses,
  stop_reasont stop_reason,
  bool reachability_trace = false)
{
  return report(options, result, witnesses, stop_reason, reachability_trace);
}
} // namespace

TEST_CASE("result-only mode prints no trace at all", "[bmc][witness]")
{
  CHECK(report(
          flags({"result-only"}),
          P_SATISFIABLE,
          {witness({1})},
          stop_reasont::Disabled)
          .empty());
}

TEST_CASE("an undischarged claim holds only up to k", "[bmc][witness]")
{
  const std::string logged =
    report(flags({}), P_UNSATISFIABLE, {}, stop_reasont::Disabled);
  CHECK_THAT(
    logged,
    Catch::Matchers::Contains(
      "Claim 'claim.assertion.1' holds up to the current K"));
}

TEST_CASE("a claim the solver could not decide says so", "[bmc][witness]")
{
  CHECK_THAT(
    report(flags({}), P_ERROR, {}, stop_reasont::Disabled),
    Catch::Matchers::Contains("Claim 'claim.assertion.1' could not be solved"));
}

TEST_CASE("a single witness keeps the counterexample form", "[bmc][witness]")
{
  const std::string logged =
    report(flags({}), P_SATISFIABLE, {witness({1})}, stop_reasont::Disabled);
  CHECK_THAT(logged, Catch::Matchers::Contains("[Counterexample]"));
  CHECK_THAT(logged, !Catch::Matchers::Contains("Summary:"));
}

TEST_CASE("a reachability run is not a violation", "[bmc][witness]")
{
  const std::string logged = report(
    flags({}), P_SATISFIABLE, {witness({1})}, stop_reasont::Disabled, true);
  CHECK_THAT(logged, Catch::Matchers::Contains("[Reachability trace]"));
  CHECK_THAT(logged, !Catch::Matchers::Contains("[Counterexample]"));
}

TEST_CASE("several witnesses are boxed and summarised", "[bmc][witness]")
{
  const std::string logged = report(
    flags({}),
    P_SATISFIABLE,
    {witness({1}), witness({2})},
    stop_reasont::Unsat);

  CHECK_THAT(logged, Catch::Matchers::Contains("[Counterexamples - 2 witness"));
  CHECK_THAT(logged, Catch::Matchers::Contains("Witness 1 of 2"));
  CHECK_THAT(logged, Catch::Matchers::Contains("Witness 2 of 2"));
  CHECK_THAT(
    logged,
    Catch::Matchers::Contains(
      "Summary: 2 distinct input tuples violate this property"));
  CHECK_THAT(logged, Catch::Matchers::Contains("UNSAT after 2 witnesses"));
}

TEST_CASE("a truncated enumeration says so up front", "[bmc][witness]")
{
  // The same fact reaches the footer, but that sits after every witness
  // block, which on a real program is tens of kilobytes further down.
  const std::string logged = report(
    flags({}),
    P_SATISFIABLE,
    {witness({1}), witness({2})},
    stop_reasont::CapHit);

  CHECK_THAT(
    logged,
    Catch::Matchers::Contains(
      "NOTE: --max-witnesses cap reached; more witnesses may exist."));
  CHECK_THAT(logged, Catch::Matchers::Contains("--max-witnesses cap reached)"));
}

TEST_CASE("an exhausted enumeration is distinguished", "[bmc][witness]")
{
  CHECK_THAT(
    report(
      flags({}),
      P_SATISFIABLE,
      {witness({1}), witness({2})},
      stop_reasont::NoInputs),
    Catch::Matchers::Contains("no enumerable nondet inputs"));
  CHECK_THAT(
    report(
      flags({}),
      P_SATISFIABLE,
      {witness({1}), witness({2})},
      stop_reasont::Error),
    Catch::Matchers::Contains("solver returned error/unknown"));
}

TEST_CASE("inputs are collected ahead of the traces", "[bmc][witness]")
{
  const std::string logged = report(
    flags({}),
    P_SATISFIABLE,
    {witness({7}), witness({8})},
    stop_reasont::Unsat);

  const size_t inputs = logged.find("Inputs by witness:");
  CHECK(inputs != std::string::npos);
  // Ahead of the first witness box, which is the whole point of collecting
  // them: they are what differs between witnesses.
  CHECK(inputs < logged.find("Witness 1 of 2"));
}

TEST_CASE("a witness with no inputs says none", "[bmc][witness]")
{
  const std::string logged = report(
    flags({}), P_SATISFIABLE, {witness({}), witness({})}, stop_reasont::Unsat);
  CHECK_THAT(logged, Catch::Matchers::Contains("Inputs : (none)"));
  CHECK_THAT(logged, !Catch::Matchers::Contains("Inputs by witness:"));
}

TEST_CASE("an incremental run names the unwinding", "[bmc][witness]")
{
  // Without it a reader cannot tell which k produced a block, or that two
  // blocks are different unwindings rather than a repeat.
  optionst options = flags({"incremental-bmc"});
  options.set_option("unwind", "3");
  CHECK_THAT(
    report(
      options,
      P_SATISFIABLE,
      {witness({1}), witness({2})},
      stop_reasont::Unsat),
    Catch::Matchers::Contains("at k = 3"));
}

TEST_CASE("reachability traces reach rather than violate", "[bmc][witness]")
{
  CHECK_THAT(
    report(
      flags({}),
      P_SATISFIABLE,
      {witness({1}), witness({2})},
      stop_reasont::Unsat,
      true),
    Catch::Matchers::Contains(
      "Summary: 2 distinct input tuples reach this goal"));
}
