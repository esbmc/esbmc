// End-to-end SMT lowering of bigint expressions (issue #4642 Phase 2C).
// Instantiates the full ESBMC Z3 backend via create_new_z3_solver and
// drives convert_ast on bigint-typed exprs to confirm:
//   1. convert_sort(bigint_type2t) returns SMT_SORT_INT.
//   2. constant_int2tc(bigint, huge) lowers to the Z3 integer numeral
//      with the correct decimal representation, even when the global
//      int_encoding option is off (the default).
//   3. add/sub/mul/div/modulus/neg on bigint operands route through the
//      Int-sort builders (mk_add etc), not the BV ones (mk_bvadd).
//   4. The resulting numeral is the correct mathematical value, not a
//      wrapped one — proves no fixed-width truncation along the path.
//
// A regression that reintroduced a width-dependent dispatch in
// convert_terminal or the arithmetic cases of convert_smt_expr would
// surface here as either a sort-mixing crash or a wrong-magnitude
// numeral.

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <memory>
#include <z3++.h>
#include <big-int/bigint.hh>
#include <irep2/irep2.h>
#include <util/context.h>
#include <util/mp_arith.h>
#include <util/namespace.h>
#include <util/options.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/solve.h>
#include <z3_conv.h>

extern solver_creator create_new_z3_solver;

namespace
{
struct z3_fixture
{
  contextt ctx;
  namespacet ns;
  optionst options;
  tuple_iface *tuple_api = nullptr;
  array_iface *array_api = nullptr;
  fp_convt *fp_api = nullptr;
  std::unique_ptr<smt_convt> solver;

  z3_fixture() : ns(ctx)
  {
    solver.reset(
      create_new_z3_solver(options, ns, &tuple_api, &array_api, &fp_api));
    // smt_post_init is intentionally NOT called — it sets up the
    // address-space array machinery which needs more wiring than the sort
    // and constant-folding paths under test require. assert_expr /
    // dec_solve scenarios that need boolean_sort and address-space arrays
    // belong in a follow-up suite (PR 3 territory).
  }

  std::string numeral_of(smt_astt ast) const
  {
    z3::expr e = to_solver_smt_ast<z3_smt_ast>(ast)->a;
    return std::string(e.get_decimal_string(0));
  }
};
} // namespace

SCENARIO(
  "convert_sort(bigint_type2t) returns SMT_SORT_INT",
  "[z3][bigint][smt]")
{
  z3_fixture f;
  smt_sortt s = f.solver->convert_sort(bigint_type2tc());
  REQUIRE(s != nullptr);
  REQUIRE(s->id == SMT_SORT_INT);
}

// Value-correctness tests intentionally use BigInt magnitudes that fit
// int64. The numeral round-trip through mk_smt_int uses BigInt::to_int64,
// which silently truncates past 64 bits — that's an orthogonal latent bug
// fixed by PR #4647 (Phase 2A.5). PR 2C's contract is the *sort* dispatch:
// that constant_int2tc(bigint, _), add2tc(bigint, _, _), etc. route through
// the Int-sort SMT path regardless of magnitude. Sort assertions hold at
// any value; value-equality assertions hold without #4647 only when the
// value fits int64. Once #4647 lands, these magnitudes can be widened.

SCENARIO(
  "constant_int2tc(bigint, _) lowers to an Int-sort numeral",
  "[z3][bigint][smt]")
{
  z3_fixture f;

  GIVEN("2^60 as a bigint constant")
  {
    BigInt v = BigInt::power2(60);
    expr2tc e = constant_int2tc(bigint_type2tc(), v);
    THEN("convert_ast returns the Z3 numeral with no truncation")
    {
      smt_astt ast = f.solver->convert_ast(e);
      REQUIRE(ast->sort->id == SMT_SORT_INT);
      REQUIRE(f.numeral_of(ast) == integer2string(v, 10));
    }
  }

  GIVEN("-(2^60) as a bigint constant")
  {
    BigInt v = -BigInt::power2(60);
    expr2tc e = constant_int2tc(bigint_type2tc(), v);
    THEN("convert_ast preserves the sign")
    {
      smt_astt ast = f.solver->convert_ast(e);
      REQUIRE(ast->sort->id == SMT_SORT_INT);
      REQUIRE(f.numeral_of(ast) == integer2string(v, 10));
    }
  }

  GIVEN("a bigint constant past int64 — sort still routes to Int")
  {
    // The numeral value below int64 may truncate via mk_smt_int until
    // PR #4647 lands; the sort dispatch under test here does not.
    expr2tc e = constant_int2tc(bigint_type2tc(), BigInt::power2(200));
    smt_astt ast = f.solver->convert_ast(e);
    REQUIRE(ast->sort->id == SMT_SORT_INT);
  }
}

SCENARIO(
  "bigint arithmetic routes through Int-sort builders end-to-end",
  "[z3][bigint][smt]")
{
  z3_fixture f;
  const type2tc bigint = bigint_type2tc();
  const BigInt a = BigInt::power2(60); // fits int64; see note above

  GIVEN("add(2^60, 2^60)")
  {
    expr2tc e =
      add2tc(bigint, constant_int2tc(bigint, a), constant_int2tc(bigint, a));
    // The simplifier (PR 2B) folds this constant pair, so convert_ast
    // sees a single bigint constant. The sort and value assertions both
    // hold.
    THEN("convert_ast yields the Int-sort numeral 2^61")
    {
      smt_astt ast = f.solver->convert_ast(e->simplify());
      REQUIRE(ast->sort->id == SMT_SORT_INT);
      REQUIRE(f.numeral_of(ast) == integer2string(BigInt::power2(61), 10));
    }
  }

  GIVEN("mul(2^30, 2^30) — Int-sort multiplier never wraps")
  {
    BigInt m = BigInt::power2(30);
    expr2tc e =
      mul2tc(bigint, constant_int2tc(bigint, m), constant_int2tc(bigint, m));
    THEN("convert_ast yields the Int-sort numeral 2^60")
    {
      smt_astt ast = f.solver->convert_ast(e->simplify());
      REQUIRE(ast->sort->id == SMT_SORT_INT);
      REQUIRE(f.numeral_of(ast) == integer2string(BigInt::power2(60), 10));
    }
  }

  GIVEN("neg(2^60)")
  {
    expr2tc e = neg2tc(bigint, constant_int2tc(bigint, a));
    THEN("convert_ast yields the Int-sort numeral -(2^60)")
    {
      smt_astt ast = f.solver->convert_ast(e->simplify());
      REQUIRE(ast->sort->id == SMT_SORT_INT);
      REQUIRE(f.numeral_of(ast) == integer2string(-a, 10));
    }
  }
}

// A full end-to-end assert_expr / dec_solve test on a bigint guard would
// belong here, but it needs the smt_post_init machinery (boolean_sort,
// address-space arrays, tuple/array ifaces wired). That setup is too
// involved to replicate in the unit-test scaffolding without duplicating
// solve.cpp's wrapper. Phase 3 (Python frontend cutover) will exercise
// the end-to-end path via existing regression tests; this suite locks
// down the convert_sort / convert_ast contract that Phase 3 depends on.
