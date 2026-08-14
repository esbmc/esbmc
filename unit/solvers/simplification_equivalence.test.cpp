// SMT equivalence checker for simplifier rewrites (issue #4625).
//
// The checker is what would tell us a simplifier rewrite changed meaning, so
// it has to be shown to say "differs" on a rewrite that really does. Driving
// it directly is the only way to get that: the installed checker aborts, and a
// regression test cannot assert on a build that aborts by design.
//
// Each case states a (before, after) pair whose relationship is decided by
// hand, then asks the checker to agree.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2_utils.h>
#include <solvers/smt/simplification_equivalence.h>
#include <util/arith/arith_tools.h>
#include <util/arith/ieee_float.h>
#include <util/config/config.h>
#include <util/config/options.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>

namespace
{
expr2tc int_symbol(const char *name)
{
  return symbol2tc(get_int32_type(), name);
}
} // namespace

SCENARIO(
  "the simplifier equivalence checker decides real rewrites",
  "[solvers][simplifier]")
{
  config.ansi_c.set_data_model(configt::LP64);
  contextt ctx;
  namespacet ns(ctx);
  optionst options;

  const expr2tc x = int_symbol("x");
  const expr2tc zero = gen_zero(get_int32_type());
  const expr2tc one = from_integer(1, get_int32_type());

  GIVEN("a meaning-preserving rewrite")
  {
    // x + 0 -> x, for every x.
    const expr2tc before = add2tc(get_int32_type(), x, zero);
    THEN("it reports equivalent")
    {
      REQUIRE(
        check_simplification_equivalence(before, x, ns, options) ==
        simplification_equivalencet::equivalent);
    }
  }

  GIVEN("a rewrite that changes meaning")
  {
    // x + 1 -> x differs at every x, so any model is a witness.
    const expr2tc before = add2tc(get_int32_type(), x, one);
    THEN("it reports differs, with a witness naming the free symbols")
    {
      std::string witness;
      simplification_equivalence_checkert checker(ns, options);
      REQUIRE(
        checker.check(before, x, &witness) ==
        simplification_equivalencet::differs);
      // The abort message is only actionable if it says on what value the two
      // disagree, so the model must actually come back.
      REQUIRE(witness.find("x") != std::string::npos);
    }
  }

  GIVEN("a rewrite that is wrong only on a corner value")
  {
    // x * 2 / 2 -> x holds except where the multiply wraps, so the checker
    // must not be satisfied by the common case.
    const expr2tc two = from_integer(2, get_int32_type());
    const expr2tc before =
      div2tc(get_int32_type(), mul2tc(get_int32_type(), x, two), two);
    THEN("it still reports differs")
    {
      REQUIRE(
        check_simplification_equivalence(before, x, ns, options) ==
        simplification_equivalencet::differs);
    }
  }

  GIVEN("a boolean rewrite")
  {
    const expr2tc b = symbol2tc(get_bool_type(), "b");
    THEN("!!b -> b is equivalent")
    {
      REQUIRE(
        check_simplification_equivalence(not2tc(not2tc(b)), b, ns, options) ==
        simplification_equivalencet::equivalent);
    }
    THEN("!b -> b differs")
    {
      REQUIRE(
        check_simplification_equivalence(not2tc(b), b, ns, options) ==
        simplification_equivalencet::differs);
    }
  }

  GIVEN("a floating-point rewrite")
  {
    const type2tc double_ty = migrate_type(double_type());
    // fp.eq is not structural equality -- NaN != NaN under it, so a checker
    // built on equality2t alone calls even the identity rewrite `differs` and
    // aborts an enabled build on the first float it meets.
    const expr2tc f = symbol2tc(double_ty, "f");
    THEN("the identity rewrite is equivalent")
    {
      REQUIRE(
        check_simplification_equivalence(f, f, ns, options) ==
        simplification_equivalencet::equivalent);
    }
    THEN("if(c,f,f) -> f is equivalent")
    {
      const expr2tc c = symbol2tc(get_bool_type(), "c");
      REQUIRE(
        check_simplification_equivalence(
          if2tc(double_ty, c, f, f), f, ns, options) ==
        simplification_equivalencet::equivalent);
    }
    THEN("-0.0 -> +0.0 differs")
    {
      // fp.eq(+0.0, -0.0) holds, so only the sign check separates them. This
      // is the class of float bug the simplifier guards against by hand.
      ieee_floatt neg_zero(ieee_float_spect::double_precision());
      neg_zero.from_double(0.0);
      neg_zero.set_sign(true);
      ieee_floatt pos_zero(ieee_float_spect::double_precision());
      pos_zero.from_double(0.0);
      REQUIRE(
        check_simplification_equivalence(
          constant_floatbv2tc(neg_zero),
          constant_floatbv2tc(pos_zero),
          ns,
          options) == simplification_equivalencet::differs);
    }
  }

  GIVEN("a shape the checker declines")
  {
    THEN("a nil operand is skipped")
    {
      REQUIRE(
        check_simplification_equivalence(expr2tc(), x, ns, options) ==
        simplification_equivalencet::skipped);
    }
    THEN("a pointer-typed expression is skipped")
    {
      const expr2tc p = symbol2tc(pointer_type2tc(get_int32_type()), "p");
      REQUIRE(
        check_simplification_equivalence(p, p, ns, options) ==
        simplification_equivalencet::skipped);
    }
    THEN("a rewrite that changes type is skipped")
    {
      REQUIRE(
        check_simplification_equivalence(
          x, symbol2tc(get_int64_type(), "y"), ns, options) ==
        simplification_equivalencet::skipped);
    }
  }
}

SCENARIO(
  "one checker decides many rewrites in sequence",
  "[solvers][simplifier]")
{
  // The installed checker reuses a single solver across every rewrite in the
  // run -- a solver per rewrite leaks ~14 KB in create_solver. Reuse is only
  // safe if a pushed frame leaves nothing behind that colours the next
  // verdict, so drive alternating verdicts through one checker and require
  // each to come out as it does in isolation.
  config.ansi_c.set_data_model(configt::LP64);
  contextt ctx;
  namespacet ns(ctx);
  optionst options;

  const expr2tc x = int_symbol("x");
  const expr2tc zero = gen_zero(get_int32_type());
  const expr2tc one = from_integer(1, get_int32_type());
  const expr2tc same = add2tc(get_int32_type(), x, zero);
  const expr2tc changed = add2tc(get_int32_type(), x, one);

  simplification_equivalence_checkert checker(ns, options);
  for (int i = 0; i < 8; ++i)
  {
    REQUIRE(checker.check(same, x) == simplification_equivalencet::equivalent);
    REQUIRE(checker.check(changed, x) == simplification_equivalencet::differs);
  }

  // A declined shape resets the solver; the checker must keep working after.
  REQUIRE(checker.check(expr2tc(), x) == simplification_equivalencet::skipped);
  REQUIRE(checker.check(same, x) == simplification_equivalencet::equivalent);
}
