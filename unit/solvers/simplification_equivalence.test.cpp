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

#include <cstdio>
#include <string>
#include <irep2/irep2_utils.h>
#include <solvers/simplification_equivalence.h>
#include <util/arith/arith_tools.h>
#include <util/arith/ieee_float.h>
#include <util/config/config.h>
#include <util/config/options.h>
#include <util/symtab/context.h>
#include <util/message/message.h>
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

  GIVEN("shapes whose conversion would not return a term")
  {
    // Each of these reaches an abort() or an assert() inside convert_ast, which
    // no catch can turn back into a decline, so the screen has to refuse them
    // up front. They only start arriving once the check covers symex and the
    // per-node folds (esbmc/esbmc#7260).
    const type2tc double_ty = migrate_type(double_type());
    const expr2tc g = symbol2tc(double_ty, "g");
    const expr2tc rne =
      from_integer(ieee_floatt::ROUND_TO_EVEN, get_int32_type());

    THEN("a value-set placeholder is declined")
    {
      const expr2tc unknown = unknown2tc(get_int32_type());
      REQUIRE(
        check_simplification_equivalence(
          add2tc(get_int32_type(), unknown, zero), unknown, ns, options) ==
        simplification_equivalencet::skipped);
    }
    THEN("a round-to-nearest-away rounding mode is declined")
    {
      // Z3 and CVC5 abort on RNA; Bitwuzla does not. Refused for all of them.
      const expr2tc rna =
        from_integer(ieee_floatt::ROUND_TO_AWAY, get_int32_type());
      REQUIRE(
        check_simplification_equivalence(
          ieee_mul2tc(double_ty, g, g, rna),
          ieee_mul2tc(double_ty, g, g, rne),
          ns,
          options) == simplification_equivalencet::skipped);
    }
    THEN("a symbolic rounding mode is declined")
    {
      // convert_rounding_mode() builds an ite over every mode, RNA included.
      const expr2tc rm =
        symbol2tc(get_int32_type(), "c:@__ESBMC_rounding_mode");
      REQUIRE(
        check_simplification_equivalence(
          ieee_mul2tc(double_ty, g, g, rm),
          ieee_mul2tc(double_ty, g, g, rne),
          ns,
          options) == simplification_equivalencet::skipped);
    }
    THEN("narrowing a float to an integer is declined")
    {
      // Undefined out of range (C11 6.3.1.4p1), so there is no value to state
      // an equality about.
      REQUIRE(
        check_simplification_equivalence(
          typecast2tc(get_int32_type(), g, rne), x, ns, options) ==
        simplification_equivalencet::skipped);
    }
    THEN("a byte extract over an array is declined")
    {
      // convert_byte_extract() asserts its source is not an array.
      const type2tc arr =
        array_type2tc(get_int32_type(), from_integer(4, size_type2()), false);
      REQUIRE(
        check_simplification_equivalence(
          byte_extract2tc(get_int8_type(), symbol2tc(arr, "a"), zero, false),
          from_integer(0, get_int8_type()),
          ns,
          options) == simplification_equivalencet::skipped);
    }
  }

  GIVEN("every node kind that carries a rounding mode")
  {
    // The screen is written against rounding_mode_of()'s node list, so each
    // kind on that list has to be shown to reach it -- a missing arm declines
    // nothing and lets the Z3/CVC5 RNA abort through (#7324).
    const type2tc double_ty = migrate_type(double_type());
    const expr2tc g = symbol2tc(double_ty, "g");
    const expr2tc rne =
      from_integer(ieee_floatt::ROUND_TO_EVEN, get_int32_type());
    const expr2tc rna =
      from_integer(ieee_floatt::ROUND_TO_AWAY, get_int32_type());

    auto declines = [&](const expr2tc &before, const expr2tc &after) {
      return check_simplification_equivalence(before, after, ns, options) ==
             simplification_equivalencet::skipped;
    };

    THEN("each is declined when its rounding mode is round-to-nearest-away")
    {
      REQUIRE(declines(ieee_add2tc(double_ty, g, g, rna), g));
      REQUIRE(declines(ieee_sub2tc(double_ty, g, g, rna), g));
      REQUIRE(declines(ieee_div2tc(double_ty, g, g, rna), g));
      REQUIRE(declines(ieee_rem2tc(double_ty, g, g, rna), g));
      REQUIRE(declines(ieee_fma2tc(double_ty, g, g, g, rna), g));
      REQUIRE(declines(ieee_sqrt2tc(double_ty, g, rna), g));
      REQUIRE(declines(nearbyint2tc(double_ty, g, rna), g));
    }

    THEN("a representable rounding mode is not declined for that reason")
    {
      // Negative control: the same shapes decide, so the screen is keying on
      // the mode rather than on the node kind.
      REQUIRE(
        check_simplification_equivalence(
          ieee_add2tc(double_ty, g, g, rne),
          ieee_add2tc(double_ty, g, g, rne),
          ns,
          options) == simplification_equivalencet::equivalent);
    }
  }

  GIVEN("an expression past the node budget")
  {
    // One query per fold is quadratic on a long chain, so check() bounds the
    // node count. The workflow's slice leans on that bound, and nothing else
    // exercises it.
    expr2tc chain = x;
    for (int i = 0; i < 300; ++i)
      chain = add2tc(get_int32_type(), chain, one);

    THEN("it is declined rather than proved")
    {
      REQUIRE(
        check_simplification_equivalence(chain, chain, ns, options) ==
        simplification_equivalencet::skipped);
    }

    THEN("a short chain of the same shape still decides")
    {
      expr2tc few = x;
      for (int i = 0; i < 4; ++i)
        few = add2tc(get_int32_type(), few, one);
      REQUIRE(
        check_simplification_equivalence(few, few, ns, options) ==
        simplification_equivalencet::equivalent);
    }
  }

  GIVEN("shapes whose bitvector lowering cannot be trusted")
  {
    THEN("an overflow_cast is declined")
    {
      // Its lowering builds an upper bound its own type cannot hold (#7324).
      REQUIRE(
        check_simplification_equivalence(
          overflow_cast2tc(x, 32), gen_false_expr(), ns, options) ==
        simplification_equivalencet::skipped);
    }

    THEN("a fixedbv constant wider than 64 bits is declined")
    {
      // convert_terminal() asserts such a constant fits a uint64_t.
      fixedbv_spect wide_spec(128, 64);
      fixedbvt wide(wide_spec);
      wide.from_integer(BigInt(1));
      const expr2tc wide_const = constant_fixedbv2tc(wide);
      REQUIRE(
        check_simplification_equivalence(wide_const, wide_const, ns, options) ==
        simplification_equivalencet::skipped);
    }
  }

  GIVEN("the unsound folds from issue #7260")
  {
    // Both were planted by hand in do_simplify() to show the hook was not
    // seeing per-node peepholes. Keeping them here pins the other half of that
    // claim: that the checker does decide them, so a widened hook has
    // something to widen onto.
    THEN("x * 2 -> x differs")
    {
      REQUIRE(
        check_simplification_equivalence(
          mul2tc(get_int32_type(), x, from_integer(2, get_int32_type())),
          x,
          ns,
          options) == simplification_equivalencet::differs);
    }
    THEN("x * 0.0 -> 0.0 differs")
    {
      // Unsound under IEEE 754 twice over: inf * 0 and NaN * 0 are NaN, and
      // the product's zero takes the sign of neither operand alone.
      const type2tc double_ty = migrate_type(double_type());
      ieee_floatt zero_val(ieee_float_spect::double_precision());
      zero_val.from_double(0.0);
      const expr2tc fp_zero = constant_floatbv2tc(zero_val);
      const expr2tc g = symbol2tc(double_ty, "g");
      const expr2tc rm =
        from_integer(ieee_floatt::ROUND_TO_EVEN, get_int32_type());
      REQUIRE(
        check_simplification_equivalence(
          ieee_mul2tc(double_ty, g, fp_zero, rm), fp_zero, ns, options) ==
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

SCENARIO(
  "the checker screens operand sorts before converting",
  "[solvers][simplifier]")
{
  // irep2 builds a relation without checking that its operands share a width
  // (irep2_expr.h gives the relational constructors an empty body), but every
  // backend's mk_eq asserts they do. An assert is abort(), so before #7220 the
  // pairs below took the whole process down instead of reaching a verdict.
  config.ansi_c.set_data_model(configt::LP64);
  contextt ctx;
  namespacet ns(ctx);
  optionst options;

  const expr2tc x = int_symbol("x");
  const expr2tc zero32 = gen_zero(get_int32_type());
  const expr2tc zero64 = gen_zero(get_int64_type());
  const expr2tc mixed_width = equality2tc(x, zero64);
  const expr2tc well_sorted = equality2tc(x, zero32);

  GIVEN("an ill-sorted subterm in `before`")
  {
    THEN("it is declined and counted apart from the other declines")
    {
      const unsigned long seen = simplification_check_stats::ill_sorted.load();
      REQUIRE(
        check_simplification_equivalence(
          mixed_width, gen_true_expr(), ns, options) ==
        simplification_equivalencet::skipped);
      REQUIRE(simplification_check_stats::ill_sorted.load() == seen + 1);
    }
    THEN("a bool compared against a 1-bit bitvector is declined too")
    {
      // bool and a 1-bit bitvector are distinct sorts however alike their bit
      // counts look, so bool has to count as widthless rather than as 1.
      const expr2tc bit = symbol2tc(unsignedbv_type2tc(1), "bit");
      REQUIRE(
        check_simplification_equivalence(
          equality2tc(gen_true_expr(), bit), gen_true_expr(), ns, options) ==
        simplification_equivalencet::skipped);
    }
  }

  GIVEN("a well-sorted `before` rewritten into an ill-sorted `after`")
  {
    THEN("it reports malformed, not a decline")
    {
      REQUIRE(
        check_simplification_equivalence(
          well_sorted, mixed_width, ns, options) ==
        simplification_equivalencet::malformed);
    }
  }

  GIVEN("a mismatch under a node other than a relation")
  {
    // operands_must_share_sort collapses twelve kinds onto one arm; drive an
    // arithmetic/bitwise one too, so narrowing the list cannot pass unnoticed.
    THEN("it is declined")
    {
      const expr2tc masked = bitand2tc(get_int32_type(), x, zero64);
      REQUIRE(
        check_simplification_equivalence(
          equality2tc(masked, x), gen_true_expr(), ns, options) ==
        simplification_equivalencet::skipped);
    }
  }

  GIVEN("an if2t whose branches disagree")
  {
    // mk_ite asserts on the branch pair exactly as mk_eq does on its operands,
    // and if2t's operands are (cond, true, false) -- so the screen has to look
    // at 1 and 2, not at 0 and 1.
    const expr2tc cond = symbol2tc(get_bool_type(), "c");
    THEN("it is declined")
    {
      const expr2tc mixed = if2tc(get_int32_type(), cond, x, zero64);
      REQUIRE(
        check_simplification_equivalence(
          equality2tc(mixed, x), gen_true_expr(), ns, options) ==
        simplification_equivalencet::skipped);
    }
    THEN("matching branches are still decided")
    {
      const expr2tc same = if2tc(get_int32_type(), cond, x, x);
      REQUIRE(
        check_simplification_equivalence(same, x, ns, options) ==
        simplification_equivalencet::equivalent);
    }
  }

  GIVEN("a float compared against a bitvector of the same width")
  {
    // Equal widths, different sorts: only a float pair takes the fp.eq path,
    // so the width assertion alone would let this through.
    THEN("it is declined")
    {
      const expr2tc f = symbol2tc(float_type2(), "f");
      REQUIRE(
        check_simplification_equivalence(
          equality2tc(f, x), gen_true_expr(), ns, options) ==
        simplification_equivalencet::skipped);
    }
  }

  GIVEN("a shift whose count is narrower than the value")
  {
    // irep2 allows this and convert_ast() casts the count up, so shifts must
    // stay out of operands_must_share_sort -- declining them would quietly
    // drop every narrow-count shift from the check's coverage.
    THEN("it is decided, not declined")
    {
      const expr2tc shifted =
        shl2tc(get_int32_type(), x, symbol2tc(get_uint8_type(), "n"));
      REQUIRE(
        check_simplification_equivalence(shifted, shifted, ns, options) ==
        simplification_equivalencet::equivalent);
    }
  }

  GIVEN("a well-sorted relation nested inside the pair")
  {
    // The screen must not decline what it can decide: x + 0 == x is true.
    THEN("it is still decided by the solver")
    {
      const expr2tc before =
        equality2tc(add2tc(get_int32_type(), x, zero32), x);
      REQUIRE(
        check_simplification_equivalence(
          before, gen_true_expr(), ns, options) ==
        simplification_equivalencet::equivalent);
    }
  }

  GIVEN("a nil operand buried under a node the screen walks through")
  {
    // and2t/not2t are not in the shared-sort list, so the recursion descends
    // into them -- and foreach_operand hands the delegate nil fields too.
    THEN("the walk survives it and still declines on the ill-sorted sibling")
    {
      const unsigned long seen = simplification_check_stats::ill_sorted.load();
      const expr2tc before = and2tc(not2tc(expr2tc()), mixed_width);
      REQUIRE(
        check_simplification_equivalence(
          before, gen_true_expr(), ns, options) ==
        simplification_equivalencet::skipped);
      REQUIRE(simplification_check_stats::ill_sorted.load() == seen + 1);
    }
  }

  GIVEN("a decline for a reason other than sorts")
  {
    // Negative control on the counter: without this, an increment moved to the
    // generic skip path would go unnoticed and the signal would mean nothing.
    THEN("the ill-sorted counter stays put")
    {
      const unsigned long seen = simplification_check_stats::ill_sorted.load();
      const expr2tc p = symbol2tc(pointer_type2tc(get_int32_type()), "p");
      REQUIRE(
        check_simplification_equivalence(p, p, ns, options) ==
        simplification_equivalencet::skipped);
      REQUIRE(simplification_check_stats::ill_sorted.load() == seen);
    }
  }
}

SCENARIO(
  "the equivalence check reports what it decided",
  "[solvers][simplifier]")
{
  // The counter only earns its keep if the summary actually separates it from
  // the other declines, so pin the rendered line rather than the call.
  const unsigned long saved_proved = simplification_check_stats::proved.load();
  const unsigned long saved_declined =
    simplification_check_stats::declined.load();
  const unsigned long saved_ill = simplification_check_stats::ill_sorted.load();

  simplification_check_stats::proved = 7;
  simplification_check_stats::declined = 5;
  simplification_check_stats::ill_sorted = 3;

  FILE *const saved_out = messaget::state.out;
  FILE *const captured = tmpfile();
  REQUIRE(captured != nullptr);
  messaget::state.out = captured;
  simplification_check_stats::report();
  messaget::state.out = saved_out;

  std::rewind(captured);
  std::string text;
  char buf[256];
  for (size_t n; (n = std::fread(buf, 1, sizeof(buf), captured)) > 0;)
    text.append(buf, n);
  std::fclose(captured);

  simplification_check_stats::proved = saved_proved;
  simplification_check_stats::declined = saved_declined;
  simplification_check_stats::ill_sorted = saved_ill;

  REQUIRE(
    text.find("7 rewrites proved, 5 declined (3 ill-sorted)") !=
    std::string::npos);
}
