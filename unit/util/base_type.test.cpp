/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Regression test for a SIGSEGV in base_type(expr2tc&).
 *
 * sideeffect2t with kind `nondet` is the canonical IR shape emitted by
 * gen_nondet() (see src/irep2/irep2_utils.h): it has nil `operand` and
 * nil `size` sub-expressions by design. Any caller that walks sub-expr
 * operands must therefore tolerate nil children, the same way check_rec
 * in goto_check and base_type_eqt::base_type_eq_rec already do.
 *
 * Before the fix, base_type(expr2tc&) dereferenced these nil children in
 * its recursive Foreach_operand walk, so a div-by-zero check whose
 * divisor contained a nondet sideeffect crashed the frontend during
 * goto_check.
 */

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/namespace.h>
#include <util/symbol.h>

TEST_CASE(
  "base_type tolerates nil sub-operands in sideeffect nondet",
  "[core][util][base_type]")
{
  contextt ctx;
  namespacet ns(ctx);

  const type2tc u32 = get_uint32_type();

  // gen_nondet() constructs a sideeffect2t whose `operand` and `size`
  // sub-expr2tc fields are intentionally nil.
  expr2tc nondet = gen_nondet(u32);
  REQUIRE(is_sideeffect2t(nondet));

  // Direct invocation on the nondet sideeffect must not deref the nil
  // children. Pre-fix this crashed with a SIGSEGV inside Foreach_operand.
  REQUIRE_NOTHROW(base_type(nondet, ns));

  // It must also work when the nondet sits inside another expression,
  // which is how goto_check's div_by_zero_check triggered the crash in
  // the wild (notequal(divisor, 0) where divisor was a nondet sideeffect).
  expr2tc zero = gen_zero(u32);
  expr2tc ne = notequal2tc(nondet, zero);
  REQUIRE_NOTHROW(base_type(ne, ns));
}

TEST_CASE(
  "base_type is a no-op on a nil top-level expression",
  "[core][util][base_type]")
{
  contextt ctx;
  namespacet ns(ctx);

  expr2tc nil_expr;
  REQUIRE(is_nil_expr(nil_expr));
  REQUIRE_NOTHROW(base_type(nil_expr, ns));
  REQUIRE(is_nil_expr(nil_expr));
}
