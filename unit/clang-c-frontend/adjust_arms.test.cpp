/*******************************************************************
 Module: clang_c_adjust_irep2 dispatch-chain unit tests

 Test Plan:
   - the chain's order constraints, which used to be statement positions
   - the guards partition the families two arms share
   - one arm's rewrite, driven through the pass's own interface
 \*******************************************************************/

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <clang-c-frontend/clang_c_adjust_irep2.h>
#include <irep2/irep2.h>
#include <irep2/irep2_utils.h>
#include <util/config/config.h>
#include <util/lang/c_types.h>
#include <util/symtab/context.h>

#include <algorithm>
#include <string>
#include <vector>

namespace
{
using arm_info = clang_c_adjust_irep2::arm_info;

std::vector<arm_info> chain()
{
  return clang_c_adjust_irep2::arm_order();
}

/// Position of \p name in the chain. Fails the test when there is no such arm,
/// so renaming an arm without updating its constraint is a failure rather than
/// a silently vacuous pass.
std::size_t position_of(const std::string &name)
{
  const std::vector<arm_info> arms = chain();
  const auto it =
    std::find_if(arms.begin(), arms.end(), [&name](const arm_info &a) {
      return name == a.name;
    });
  REQUIRE(it != arms.end());
  return static_cast<std::size_t>(it - arms.begin());
}

/// The arms whose guard claims \p expr, by name.
std::vector<std::string> claimants(const expr2tc &expr)
{
  std::vector<std::string> names;
  for (const arm_info &a : chain())
    if (a.when != nullptr && a.when(expr))
      names.push_back(a.name);
  return names;
}

bool claimed_by(const expr2tc &expr, const std::string &name)
{
  const std::vector<std::string> names = claimants(expr);
  return std::find(names.begin(), names.end(), name) != names.end();
}
} // namespace

SCENARIO(
  "the IREP2 adjuster's arm order is data a test can read",
  "[core][clang-c-frontend][irep2-adjust]")
{
  GIVEN("the chain")
  {
    THEN("the designator sugar is installed before the callee is read")
    {
      // adjust_call_callee decides whether a call is direct by reading the
      // sugar adjust_function_designators installs.
      REQUIRE(
        position_of("adjust_function_designators") <
        position_of("adjust_call_callee"));
    }

    THEN("a loop's guard is converted before the for is hoisted")
    {
      // hoist_for_init rewrites a code_for2t into a block, and a block is not
      // a statement-with-condition, so the guard would never reach the
      // conversion if the hoist ran first.
      REQUIRE(
        position_of("adjust_statement_condition") <
        position_of("hoist_for_init"));
    }

    THEN("the self-gating arm runs first and the address decay runs last")
    {
      // adjust_function_designators installs sugar every later arm may read, so
      // nothing may be inserted before it. adjust_address_of ran after the
      // chain returned, so it stays last.
      const std::vector<arm_info> arms = chain();
      REQUIRE(std::string(arms.front().name) == "adjust_function_designators");
      REQUIRE(std::string(arms.back().name) == "adjust_address_of");
    }

    THEN("every arm is named exactly once")
    {
      std::vector<std::string> names;
      for (const arm_info &a : chain())
        names.push_back(a.name);
      std::sort(names.begin(), names.end());
      REQUIRE(std::adjacent_find(names.begin(), names.end()) == names.end());
    }
  }
}

SCENARIO(
  "the IREP2 adjuster's guards partition the families two arms share",
  "[core][clang-c-frontend][irep2-adjust]")
{
  GIVEN("a complex-typed negation")
  {
    const type2tc cplx = complex_type2tc(float_type2());
    const expr2tc z = symbol2tc(cplx, "z");
    const expr2tc neg = neg2tc(cplx, z);

    THEN("the complex arm claims it and the promotion arm does not")
    {
      // This was an `else` in the chain: is_complex_unary is a strict subset
      // of neg||bitnot, so the exclusion is real and now lives in the guard.
      REQUIRE(claimed_by(neg, "adjust_complex_unary"));
      REQUIRE_FALSE(claimed_by(neg, "promote_unary_bool_operand"));
    }
  }

  GIVEN("a plain integer negation")
  {
    const type2tc i32 = get_int_type(32);
    const expr2tc x = symbol2tc(i32, "x");
    const expr2tc neg = neg2tc(i32, x);

    THEN("the promotion arm claims it and the complex arm does not")
    {
      REQUIRE(claimed_by(neg, "promote_unary_bool_operand"));
      REQUIRE_FALSE(claimed_by(neg, "adjust_complex_unary"));
    }
  }

  GIVEN("a left shift")
  {
    const type2tc i32 = get_int_type(32);
    const expr2tc x = symbol2tc(i32, "x");
    const expr2tc shifted = shl2tc(i32, x, constant_int2tc(i32, BigInt(1)));

    THEN("the shift arm claims it and the arithmetic arm does not")
    {
      // The chain spelled this as an `else if`, but the two guards are
      // disjoint by expr_id: the `else` was an optimisation, not semantics.
      // Pin the disjointness so adding a shift kind to is_arith_or_bitwise
      // fails here rather than converting the node twice.
      REQUIRE(claimed_by(shifted, "adjust_shift_operands"));
      REQUIRE_FALSE(claimed_by(shifted, "adjust_binary_arith_operands"));
    }
  }
}

SCENARIO(
  "the IREP2 adjuster retypes a comma expression",
  "[core][clang-c-frontend][irep2-adjust]")
{
  GIVEN("a comma whose type disagrees with its right operand")
  {
    contextt ctx;
    clang_c_adjust_irep2 pass(ctx, true);

    const type2tc i32 = get_int_type(32);
    const type2tc i64 = get_int_type(64);
    const expr2tc lhs = symbol2tc(i32, "c");
    const expr2tc rhs = symbol2tc(i64, "a");
    expr2tc comma = code_comma2tc(i32, lhs, rhs);

    WHEN("the pass adjusts it")
    {
      // Driven through adjust_expr, the pass's own interface -- the arm is not
      // reachable by name and does not need to be. C11 6.5.17p2: a comma takes
      // its right operand's type.
      pass.adjust_expr(comma);

      THEN("it takes the right operand's type")
      {
        REQUIRE(comma->type == i64);
      }
    }
  }
}

SCENARIO(
  "the IREP2 adjuster decodes both call shapes and rewrites the callee",
  "[core][clang-c-frontend][irep2-adjust]")
{
  // A no-argument function, and the implicit `&f` designator sugar the frontend
  // installs on a direct call. adjust_call_callee strips it back to a direct f.
  const type2tc ftype = code_type2tc(
    std::vector<type2tc>{}, get_empty_type(), std::vector<irep_idt>{}, false);
  const expr2tc f = symbol2tc(ftype, "f");
  const expr2tc sugar = address_of2tc(ftype, f, /*is_implicit=*/true);

  GIVEN("a code_function_call2t whose callee is the implicit &f sugar")
  {
    contextt ctx;
    clang_c_adjust_irep2 pass(ctx, true);
    expr2tc call =
      code_function_call2tc(expr2tc(), sugar, std::vector<expr2tc>{});

    WHEN("the pass adjusts it")
    {
      // Driven through adjust_expr, the pass's own interface. as_call reads the
      // callee from the code_function_call2t's .function slot and writes it
      // back there; the implicit &f collapses to a direct f.
      pass.adjust_expr(call);

      THEN("the callee is decoded from .function and the sugar stripped")
      {
        REQUIRE(to_code_function_call2t(call).function == f);
      }
    }
  }

  GIVEN("a sideeffect2t of kind function_call with the same &f sugar")
  {
    contextt ctx;
    clang_c_adjust_irep2 pass(ctx, true);
    expr2tc call = sideeffect2tc(
      get_empty_type(),
      sugar,
      expr2tc(),
      std::vector<expr2tc>{},
      type2tc(),
      sideeffect_allockind::function_call);

    WHEN("the pass adjusts it")
    {
      // The other spelling: the callee lives in .operand, not .function, and
      // as_call must read and write the same node through that slot.
      pass.adjust_expr(call);

      THEN("the callee is decoded from .operand and the sugar stripped")
      {
        REQUIRE(to_sideeffect2t(call).operand == f);
      }
    }
  }

  GIVEN("a sideeffect2t of a non-call kind through a function pointer")
  {
    contextt ctx;
    clang_c_adjust_irep2 pass(ctx, true);
    const expr2tc fp = symbol2tc(pointer_type2tc(ftype), "fp");
    expr2tc se = sideeffect2tc(
      get_int_type(32),
      fp,
      expr2tc(),
      std::vector<expr2tc>{},
      type2tc(),
      sideeffect_allockind::nondet);

    WHEN("the pass adjusts it")
    {
      // as_call's kind gate rejects any sideeffect2t that is not a call, so
      // adjust_call_callee leaves the pointer-typed operand undereferenced.
      pass.adjust_expr(se);

      THEN("the callee slot is untouched")
      {
        REQUIRE(to_sideeffect2t(se).operand == fp);
      }
    }
  }

  // The other write-back branch: a pointer-typed callee is dereferenced in
  // place, through the same *call->callee slot, for both shapes.
  const expr2tc fp = symbol2tc(pointer_type2tc(ftype), "fp");

  GIVEN("a code_function_call2t through a function pointer")
  {
    contextt ctx;
    clang_c_adjust_irep2 pass(ctx, true);
    expr2tc call = code_function_call2tc(expr2tc(), fp, std::vector<expr2tc>{});

    WHEN("the pass adjusts it")
    {
      pass.adjust_expr(call);

      THEN("the .function slot is dereferenced")
      {
        const expr2tc &callee = to_code_function_call2t(call).function;
        REQUIRE(is_dereference2t(callee));
        REQUIRE(to_dereference2t(callee).value == fp);
      }
    }
  }

  GIVEN("a sideeffect2t of kind function_call through a function pointer")
  {
    contextt ctx;
    clang_c_adjust_irep2 pass(ctx, true);
    expr2tc call = sideeffect2tc(
      get_empty_type(),
      fp,
      expr2tc(),
      std::vector<expr2tc>{},
      type2tc(),
      sideeffect_allockind::function_call);

    WHEN("the pass adjusts it")
    {
      pass.adjust_expr(call);

      THEN("the .operand slot is dereferenced")
      {
        const expr2tc &callee = to_sideeffect2t(call).operand;
        REQUIRE(is_dereference2t(callee));
        REQUIRE(to_dereference2t(callee).value == fp);
      }
    }
  }
}

int main(int argc, char *argv[])
{
  // c_typecastt ranks operands against config.ansi_c, which is zero-initialised
  // in a unit binary. Pin a model in main() rather than at namespace scope:
  // `config` lives in another translation unit, so a static initialiser here
  // would race its constructor (unit/util/c_typecast.test.cpp does the same).
  config.ansi_c.set_data_model(configt::LP64);
  return Catch::Session().run(argc, argv);
}
