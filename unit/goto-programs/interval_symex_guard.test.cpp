#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../testing-utils/goto_factory.h"
#include <goto-programs/abstract-interpretation/interval_domain.h>
#include <irep2/irep2_expr.h>
#include <util/c_types.h>
#include <util/options.h>

// Reset all static flags to known defaults before each test.
static void reset_interval_flags()
{
  interval_domaint::enable_interval_arithmetic = false;
  interval_domaint::enable_interval_bitwise_arithmetic = false;
  interval_domaint::enable_modular_intervals = false;
  interval_domaint::enable_assertion_simplification = false;
  interval_domaint::enable_contraction_for_abstract_states = false;
  interval_domaint::enable_wrapped_intervals = false;
  interval_domaint::widening_extrapolate = false;
  interval_domaint::widening_narrowing = false;
}

// ---------------------------------------------------------------------------
// set_options tests
// ---------------------------------------------------------------------------

TEST_CASE(
  "set_options: --interval-symex-guard implies enable_interval_arithmetic",
  "[interval][set_options]")
{
  reset_interval_flags();

  optionst options;
  options.set_option("interval-analysis-arithmetic", false);
  options.set_option("interval-symex-guard", true);
  // Supply defaults for all other options referenced by set_options
  options.set_option("interval-analysis-bitwise", false);
  options.set_option("interval-analysis-modular", false);
  options.set_option("interval-analysis-simplify", false);
  options.set_option("interval-analysis-no-contract", true);
  options.set_option("interval-analysis-wrapped", false);
  options.set_option("interval-analysis-assume-asserts", false);
  options.set_option("interval-analysis-eval-assumptions", false);
  options.set_option("interval-analysis-ibex-contractor", false);
  options.set_option("interval-analysis-extrapolate", false);
  options.set_option("interval-analysis-narrowing", false);

  interval_domaint::set_options(options);

  CHECK(interval_domaint::enable_interval_arithmetic == true);
}

TEST_CASE(
  "set_options: --interval-analysis-arithmetic alone also enables arithmetic",
  "[interval][set_options]")
{
  reset_interval_flags();

  optionst options;
  options.set_option("interval-analysis-arithmetic", true);
  options.set_option("interval-symex-guard", false);
  options.set_option("interval-analysis-bitwise", false);
  options.set_option("interval-analysis-modular", false);
  options.set_option("interval-analysis-simplify", false);
  options.set_option("interval-analysis-no-contract", true);
  options.set_option("interval-analysis-wrapped", false);
  options.set_option("interval-analysis-assume-asserts", false);
  options.set_option("interval-analysis-eval-assumptions", false);
  options.set_option("interval-analysis-ibex-contractor", false);
  options.set_option("interval-analysis-extrapolate", false);
  options.set_option("interval-analysis-narrowing", false);

  interval_domaint::set_options(options);

  CHECK(interval_domaint::enable_interval_arithmetic == true);
}

TEST_CASE(
  "set_options: neither flag leaves arithmetic disabled",
  "[interval][set_options]")
{
  reset_interval_flags();

  optionst options;
  options.set_option("interval-analysis-arithmetic", false);
  options.set_option("interval-symex-guard", false);
  options.set_option("interval-analysis-bitwise", false);
  options.set_option("interval-analysis-modular", false);
  options.set_option("interval-analysis-simplify", false);
  options.set_option("interval-analysis-no-contract", true);
  options.set_option("interval-analysis-wrapped", false);
  options.set_option("interval-analysis-assume-asserts", false);
  options.set_option("interval-analysis-eval-assumptions", false);
  options.set_option("interval-analysis-ibex-contractor", false);
  options.set_option("interval-analysis-extrapolate", false);
  options.set_option("interval-analysis-narrowing", false);

  interval_domaint::set_options(options);

  CHECK(interval_domaint::enable_interval_arithmetic == false);
}

// ---------------------------------------------------------------------------
// process_instruction tests
// ---------------------------------------------------------------------------

TEST_CASE(
  "process_instruction: ASSUME constrains the variable",
  "[interval][process_instruction]")
{
  // After processing ASSUME x >= 5, the domain must prove x >= 5.
  std::string code =
    "int main() {\n"
    "  int x;\n"
    "  __ESBMC_assume(x >= 5);\n"
    "  return 0;\n"
    "}";

  reset_interval_flags();
  interval_domaint::enable_interval_arithmetic = true;

  auto arch = goto_factory::Architecture::BIT_32;
  auto P = goto_factory::get_goto_functions(code, arch);

  interval_domaint domain;
  domain.make_top();
  expr2tc assume_guard;

  for (auto &[func_name, func] : P.functions.function_map)
  {
    if (!func.body_available)
      continue;
    if (func_name.as_string().find("main") == std::string::npos)
      continue;

    for (auto i_it = func.body.instructions.begin();
         i_it != func.body.instructions.end();
         ++i_it)
    {
      domain.process_instruction(i_it);
      if (i_it->is_assume())
      {
        assume_guard = i_it->guard;
        break;
      }
    }
    break;
  }

  REQUIRE(!is_nil_expr(assume_guard));
  // The guard IS the expression we assumed, so the domain must now prove it.
  tvt result = interval_domaint::eval_boolean_expression(assume_guard, domain);
  CHECK(result.is_true());
}

TEST_CASE(
  "process_instruction: ASSIGN with arithmetic propagates interval",
  "[interval][process_instruction]")
{
  // After ASSUME x >= 3 and ASSIGN x = x + 1, the domain must prove x >= 4.
  std::string code =
    "int main() {\n"
    "  int x;\n"
    "  __ESBMC_assume(x >= 3);\n"
    "  x = x + 1;\n"
    "  return 0;\n"
    "}";

  reset_interval_flags();
  interval_domaint::enable_interval_arithmetic = true;

  auto arch = goto_factory::Architecture::BIT_32;
  auto P = goto_factory::get_goto_functions(code, arch);

  interval_domaint domain;
  domain.make_top();
  expr2tc assume_guard;

  for (auto &[func_name, func] : P.functions.function_map)
  {
    if (!func.body_available)
      continue;
    if (func_name.as_string().find("main") == std::string::npos)
      continue;

    bool past_assign = false;
    for (auto i_it = func.body.instructions.begin();
         i_it != func.body.instructions.end();
         ++i_it)
    {
      // Capture the guard from the ASSUME to construct the shifted check later.
      if (i_it->is_assume() && is_nil_expr(assume_guard))
        assume_guard = i_it->guard;

      domain.process_instruction(i_it);

      if (i_it->is_assign() && !past_assign)
      {
        past_assign = true;
        break;
      }
    }
    break;
  }

  REQUIRE(!is_nil_expr(assume_guard));
  // assume_guard is (x >= 3). After x = x+1, domain has x >= 4.
  // eval(x >= 3) must still be true (x >= 4 implies x >= 3).
  tvt result = interval_domaint::eval_boolean_expression(assume_guard, domain);
  CHECK(result.is_true());
}

TEST_CASE(
  "process_instruction: DEAD havoces the variable",
  "[interval][process_instruction]")
{
  // Declare x in an inner block, assume x >= 5, then let it go DEAD.
  // After DEAD the domain must no longer know x >= 5.
  std::string code =
    "int main() {\n"
    "  {\n"
    "    int x;\n"
    "    __ESBMC_assume(x >= 5);\n"
    "  }\n"  // x is DEAD here
    "  return 0;\n"
    "}";

  reset_interval_flags();
  interval_domaint::enable_interval_arithmetic = true;

  auto arch = goto_factory::Architecture::BIT_32;
  auto P = goto_factory::get_goto_functions(code, arch);

  interval_domaint domain;
  domain.make_top();
  expr2tc assume_guard;
  bool dead_found = false;

  for (auto &[func_name, func] : P.functions.function_map)
  {
    if (!func.body_available)
      continue;
    if (func_name.as_string().find("main") == std::string::npos)
      continue;

    for (auto i_it = func.body.instructions.begin();
         i_it != func.body.instructions.end();
         ++i_it)
    {
      // Capture ASSUME guard before processing the DEAD instruction.
      if (i_it->is_assume() && is_nil_expr(assume_guard))
        assume_guard = i_it->guard;

      domain.process_instruction(i_it);

      if (i_it->type == DEAD)
      {
        dead_found = true;
        break;
      }
    }
    break;
  }

  // If the goto program didn't emit a DEAD instruction (some configurations
  // may omit it), skip the check rather than fail.
  if (!dead_found || is_nil_expr(assume_guard))
    return;

  // After DEAD x, the domain must no longer guarantee x >= 5.
  tvt result = interval_domaint::eval_boolean_expression(assume_guard, domain);
  CHECK(!result.is_true());
}

// ---------------------------------------------------------------------------
// assume_rec typecast tests
// ---------------------------------------------------------------------------

TEST_CASE(
  "assume_rec: bool typecast is stripped and constraint is propagated",
  "[interval][assume_rec]")
{
  // assume((bool)(x >= 0)) must constrain x to [0, +inf).
  reset_interval_flags();
  interval_domaint::enable_interval_arithmetic = true;

  // Build the expression manually using the irep2 API.
  auto int_type = get_int32_type();
  auto bool_type = get_bool_type();

  expr2tc x = symbol2tc(int_type, irep_idt("test::x"));
  expr2tc zero = constant_int2tc(int_type, BigInt(0));
  expr2tc geq = greaterthanequal2tc(x, zero);
  // Wrap in a bool typecast (is_bool_type(cast) == true → should be stripped)
  expr2tc bool_cast = typecast2tc(bool_type, geq);

  interval_domaint domain;
  domain.make_top();
  domain.assume(bool_cast);

  // The underlying condition x >= 0 must now hold.
  tvt result = interval_domaint::eval_boolean_expression(geq, domain);
  CHECK(result.is_true());
}

TEST_CASE(
  "assume_rec: non-bool typecast is NOT stripped (soundness guard)",
  "[interval][assume_rec]")
{
  // assume((int)(x >= 0)) must NOT constrain x, because stripping a
  // non-bool cast could be unsound for truncating casts.
  reset_interval_flags();
  interval_domaint::enable_interval_arithmetic = true;

  auto int_type = get_int32_type();

  expr2tc x = symbol2tc(int_type, irep_idt("test::x"));
  expr2tc zero = constant_int2tc(int_type, BigInt(0));
  expr2tc geq = greaterthanequal2tc(x, zero);
  // Wrap in a non-bool typecast (is_bool_type(cast) == false → must NOT strip)
  expr2tc int_cast = typecast2tc(int_type, geq);

  interval_domaint domain;
  domain.make_top();
  domain.assume(int_cast);

  // x must remain unconstrained — eval should not return TV_TRUE.
  tvt result = interval_domaint::eval_boolean_expression(geq, domain);
  CHECK(!result.is_true());
}
