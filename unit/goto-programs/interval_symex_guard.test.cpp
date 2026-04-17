#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <goto-programs/abstract-interpretation/interval_domain.h>
#include <goto-programs/goto_program.h>
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
  "set_options: interval-symex-guard on by default enables interval_arithmetic",
  "[interval][set_options]")
{
  reset_interval_flags();

  optionst options;
  options.set_option("interval-analysis-arithmetic", false);
  // symex_assign sets this to true before calling set_options when feature is active
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
  "set_options: --no-interval-symex-guard without arithmetic leaves it "
  "disabled",
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
  // Construct an ASSUME instruction directly and verify the domain is updated.
  reset_interval_flags();
  interval_domaint::enable_interval_arithmetic = true;

  auto int_type = get_int32_type();
  expr2tc x = symbol2tc(int_type, irep_idt("x"));
  expr2tc five = constant_int2tc(int_type, BigInt(5));
  expr2tc geq = greaterthanequal2tc(x, five);

  // Build a minimal ASSUME instruction list so we have a valid iterator.
  goto_programt::instructionst instrs;
  instrs.emplace_back();
  auto it = instrs.begin();
  it->type = ASSUME;
  it->guard = geq;

  interval_domaint domain;
  domain.make_top();
  domain.process_instruction(it);

  // After ASSUME x >= 5 the domain must prove x >= 5.
  tvt result = interval_domaint::eval_boolean_expression(geq, domain);
  CHECK(result.is_true());
}

TEST_CASE(
  "process_instruction: ASSIGN with arithmetic propagates interval",
  "[interval][process_instruction]")
{
  // Construct ASSUME x >= 3 then ASSIGN x = x + 1; domain must prove x >= 4.
  reset_interval_flags();
  interval_domaint::enable_interval_arithmetic = true;

  auto int_type = get_int32_type();
  expr2tc x = symbol2tc(int_type, irep_idt("x"));
  expr2tc three = constant_int2tc(int_type, BigInt(3));
  expr2tc four = constant_int2tc(int_type, BigInt(4));
  expr2tc one = constant_int2tc(int_type, BigInt(1));

  goto_programt::instructionst instrs;

  // ASSUME x >= 3
  instrs.emplace_back();
  auto assume_it = instrs.begin();
  assume_it->type = ASSUME;
  assume_it->guard = greaterthanequal2tc(x, three);

  // ASSIGN x = x + 1
  instrs.emplace_back();
  auto assign_it = std::next(instrs.begin());
  assign_it->type = ASSIGN;
  assign_it->code = code_assign2tc(x, add2tc(int_type, x, one));

  interval_domaint domain;
  domain.make_top();
  domain.process_instruction(assume_it); // x ∈ [3, +∞)
  domain.process_instruction(assign_it); // x ∈ [4, +∞)

  // x >= 3 must still hold (x >= 4 implies x >= 3).
  tvt geq3 = interval_domaint::eval_boolean_expression(
    greaterthanequal2tc(x, three), domain);
  CHECK(geq3.is_true());

  // x >= 4 must now hold.
  tvt geq4 = interval_domaint::eval_boolean_expression(
    greaterthanequal2tc(x, four), domain);
  CHECK(geq4.is_true());
}

TEST_CASE(
  "process_instruction: DEAD havoces the variable",
  "[interval][process_instruction]")
{
  // Establish x >= 5 via ASSUME, then invalidate it via DEAD.
  // After DEAD the domain must no longer prove x >= 5.
  reset_interval_flags();
  interval_domaint::enable_interval_arithmetic = true;

  auto int_type = get_int32_type();
  expr2tc x = symbol2tc(int_type, irep_idt("x"));
  expr2tc five = constant_int2tc(int_type, BigInt(5));
  expr2tc geq = greaterthanequal2tc(x, five);

  goto_programt::instructionst instrs;

  // ASSUME x >= 5
  instrs.emplace_back();
  auto assume_it = instrs.begin();
  assume_it->type = ASSUME;
  assume_it->guard = geq;

  // DEAD x  (havoc_rec uses is_symbol2t to erase the entry)
  instrs.emplace_back();
  auto dead_it = std::next(instrs.begin());
  dead_it->type = DEAD;
  dead_it->code = x;

  interval_domaint domain;
  domain.make_top();
  domain.process_instruction(assume_it);

  // Before DEAD: the domain must guarantee x >= 5.
  tvt before = interval_domaint::eval_boolean_expression(geq, domain);
  REQUIRE(before.is_true());

  domain.process_instruction(dead_it);

  // After DEAD: x is havoced — the domain must not guarantee x >= 5.
  tvt after = interval_domaint::eval_boolean_expression(geq, domain);
  CHECK(!after.is_true());
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
