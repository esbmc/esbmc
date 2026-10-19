#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <cstdio>
#include <memory>
#include <string>
#include <z3_conv.h>
#include <irep2/irep2_utils.h>
#include <util/config/config.h>
#include <util/config/options.h>
#include <util/lang/c_types.h>
#include <util/message/message.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>

TEST_CASE("Z3 reports why solving was inconclusive", "[z3][diagnostics]")
{
  config.ansi_c.set_data_model(configt::LP64);
  contextt context;
  namespacet ns(context);
  optionst options;
  z3_convt solver(ns, options);
  solver.set_tuple_iface(&solver);
  solver.set_array_iface(&solver);
  solver.set_fp_conv(&solver);
  solver.smt_post_init();

  // A resource limit gives a deterministic unknown without a timing budget.
  solver.solver.set("rlimit", 1U);
  solver.assert_expr(symbol2tc(get_bool_type(), "claim"));

  const auto close_file = [](FILE *file) { std::fclose(file); };
  std::unique_ptr<FILE, decltype(close_file)> output(
    std::tmpfile(), close_file);
  REQUIRE(output != nullptr);
  struct restore_outputt
  {
    FILE *saved = messaget::state.out;
    ~restore_outputt()
    {
      messaget::state.out = saved;
    }
  } restore_output;
  messaget::state.out = output.get();

  const smt_resultt result = solver.dec_solve();
  std::rewind(output.get());
  std::string text;
  char buffer[256];
  for (size_t n; (n = std::fread(buffer, 1, sizeof(buffer), output.get())) > 0;)
    text.append(buffer, n);

  REQUIRE(result == P_ERROR);
  const std::string reason = solver.solver.reason_unknown();
  REQUIRE_FALSE(reason.empty());
  REQUIRE(
    text.find("Z3 solver returned unknown: " + reason) != std::string::npos);
}

TEST_CASE("Z3 preserves conclusive results", "[z3][diagnostics]")
{
  config.ansi_c.set_data_model(configt::LP64);
  contextt context;
  namespacet ns(context);
  optionst options;
  z3_convt solver(ns, options);
  solver.set_tuple_iface(&solver);
  solver.set_array_iface(&solver);
  solver.set_fp_conv(&solver);
  solver.smt_post_init();
  const bool value = GENERATE(false, true);
  solver.assert_expr(constant_bool2tc(value));

  REQUIRE(solver.dec_solve() == (value ? P_SATISFIABLE : P_UNSATISFIABLE));
}
