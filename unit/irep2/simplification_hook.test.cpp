// What the simplifier hook actually sees (issue #7260).
//
// The SMT checker can only prove rewrites it is handed. Before #7260 the hook
// was consulted from one place -- the free simplify(expr2tc &) wrapper -- so a
// rewrite performed by a node's own do_simplify(), or reached through
// expr2t::simplify() directly, was never offered to it. A deliberately unsound
// peephole passed unreported.
//
// These cases assert coverage rather than soundness: install a recorder, drive
// the simplifier the way ESBMC's own call sites drive it, and require the
// rewrite to show up. Proving a recorded pair right or wrong is the checker's
// job, and unit/solvers/simplification_equivalence.test.cpp's.
//
// Built only when ENABLE_SIMPLIFIER_EQUIVALENCE_CHECK is on, since that is what
// compiles verify_rewrite() into something other than a no-op.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <vector>
#include <irep2/irep2_utils.h>
#include <irep2/simplification_check.h>
#include <util/config/config.h>

namespace
{
using rewritet = std::pair<expr2tc, expr2tc>;

/** Installs a recorder for the duration of a scope; the hook is a process-wide
 *  global, so leaving one behind would leak into the next case. */
class recordert
{
public:
  recordert()
  {
    simplification_check::install(
      [this](const expr2tc &before, const expr2tc &after) {
        seen.emplace_back(before, after);
      });
  }

  ~recordert()
  {
    simplification_check::clear();
  }

  bool saw(const expr2tc &before, const expr2tc &after) const
  {
    for (const rewritet &r : seen)
      if (r.first == before && r.second == after)
        return true;
    return false;
  }

  std::vector<rewritet> seen;
};

expr2tc int_symbol(const char *name)
{
  return symbol2tc(get_int32_type(), name);
}

expr2tc int_const(int v)
{
  return constant_int2tc(get_int32_type(), BigInt(v));
}
} // namespace

SCENARIO("the simplifier hook sees per-node peepholes", "[irep2][simplifier]")
{
  config.ansi_c.set_data_model(configt::LP64);

  GIVEN("a fold that fires below the top of the expression")
  {
    const expr2tc x = int_symbol("x");
    const expr2tc y = int_symbol("y");
    const expr2tc inner = mul2tc(get_int32_type(), x, int_const(1));
    const expr2tc expr = add2tc(get_int32_type(), inner, y);

    WHEN("the expression is simplified")
    {
      recordert recorder;
      const expr2tc result = expr->simplify();

      THEN("the inner node's rewrite is offered to the checker")
      {
        // x * 1 -> x happens in mul2t::do_simplify(), reached only through
        // expr2t::simplify()'s operand walk. This is the case #7260 reports:
        // before the fix nothing at all was recorded here.
        REQUIRE(recorder.saw(inner, x));
      }

      THEN("the fold still took effect")
      {
        REQUIRE(!is_nil_expr(result));
        REQUIRE(result == add2tc(get_int32_type(), x, y));
      }
    }
  }

  GIVEN("a node reached without the free simplify() wrapper")
  {
    // arith_tools.cpp, goto_coverage.cpp and the array/vector type constructors
    // all call expr->simplify() directly. Driving it the same way here pins
    // that those entry points are covered too.
    const expr2tc x = int_symbol("x");
    const expr2tc expr = add2tc(get_int32_type(), x, int_const(0));

    WHEN("expr2t::simplify() is called on it directly")
    {
      recordert recorder;
      const expr2tc result = expr->simplify();

      THEN("its own peephole is offered to the checker")
      {
        REQUIRE(recorder.saw(expr, x));
        REQUIRE(result == x);
      }
    }
  }

  GIVEN("a chain the reassociator rewrites in place")
  {
    // Step 4 hands the checker a container it captured before the reassociators
    // ran. They mutate through non-const accessors, so copy-on-write is what
    // keeps that container pointing at the pre-rewrite tree. If it did not, the
    // pair would arrive with both sides identical and every reassociation would
    // prove itself trivially -- a silent hole rather than a failure.
    const expr2tc x = int_symbol("x");
    const expr2tc expr = add2tc(
      get_int32_type(),
      add2tc(get_int32_type(), x, int_const(1)),
      int_const(2));

    WHEN("the expression is simplified")
    {
      recordert recorder;
      const expr2tc result = expr->simplify();

      THEN("no recorded rewrite has a before that aliases its after")
      {
        REQUIRE(!recorder.seen.empty());
        for (const rewritet &r : recorder.seen)
          REQUIRE(r.first != r.second);
      }

      THEN("the constants folded")
      {
        REQUIRE(result == add2tc(get_int32_type(), x, int_const(3)));
      }
    }
  }

  GIVEN("an and/or/if node whose decisive operand short-circuits")
  {
    // The and/or/if pre-pass runs do_simplify() before the operand walk and
    // returns straight out of it -- a rewrite site the other cases never reach.
    const expr2tc b = symbol2tc(get_bool_type(), "b");
    const expr2tc no = gen_false_expr();
    const expr2tc expr = and2tc(no, b);

    WHEN("the expression is simplified")
    {
      recordert recorder;
      const expr2tc result = expr->simplify();

      THEN("the short-circuit rewrite is offered to the checker")
      {
        REQUIRE(recorder.saw(expr, no));
        REQUIRE(result == no);
      }
    }
  }
}
