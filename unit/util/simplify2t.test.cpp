/*
 * SPDX-FileCopyrightText: 2025 Lucas Cordeiro, Jeremy Morse, Bernd Fischer, Mikhail Ramalho
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/simplify_expr.h>

namespace
{
struct config_init
{
  config_init()
  {
    config.ansi_c.address_width = 64;
  }
};
const config_init init;
} // namespace

TEST_CASE("Addition simplification: x + (-x) = 0", "[arithmetic][add]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc neg_x = neg2tc(get_int_type(32), x);
  const expr2tc expr = add2tc(get_int_type(32), x, neg_x);

  const expr2tc result = expr->simplify();

  REQUIRE(!is_nil_expr(result));
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 0);
}

TEST_CASE("Addition simplification: x + 0 = x", "[arithmetic][add]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc zero = constant_int2tc(get_int_type(32), BigInt(0));

  const expr2tc expr = add2tc(get_int_type(32), x, zero);

  const expr2tc result = expr->simplify();

  REQUIRE(!is_nil_expr(result));
  REQUIRE(result == x);
}

TEST_CASE("Addition constant folding: 5 + 3 = 8", "[arithmetic][add]")
{
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc three = constant_int2tc(get_int_type(32), BigInt(3));
  const expr2tc add = add2tc(get_int_type(32), five, three);

  const expr2tc result = add->simplify();

  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 8);
}

TEST_CASE("Addition simplification: (base - X) + X = base", "[arithmetic][add]")
{
  const expr2tc base = symbol2tc(get_int_type(32), "base");
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc sub = sub2tc(get_int_type(32), base, x);

  const expr2tc add = add2tc(get_int_type(32), sub, x);

  const expr2tc result = add->simplify();

  REQUIRE(result == base);
}

TEST_CASE("Subtraction simplification: x - x = 0", "[arithmetic][sub]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");

  const expr2tc sub = sub2tc(get_int_type(32), x, x);

  const expr2tc result = sub->simplify();

  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 0);
}

TEST_CASE(
  "Reassoc: nested mul under add still canonicalizes",
  "[arithmetic][reassoc]")
{
  // x + ((a * 2) * (b * 3)) — the outer add reassoc would treat the mul
  // subtree as opaque, but the simplifier must still let the mul chain
  // canonicalize into 6*a*b before the add sees it.
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc a = symbol2tc(get_int_type(32), "a");
  const expr2tc b = symbol2tc(get_int_type(32), "b");
  const expr2tc two = constant_int2tc(get_int_type(32), BigInt(2));
  const expr2tc three = constant_int2tc(get_int_type(32), BigInt(3));
  const expr2tc a2 = mul2tc(get_int_type(32), a, two);
  const expr2tc b3 = mul2tc(get_int_type(32), b, three);
  const expr2tc m = mul2tc(get_int_type(32), a2, b3);
  const expr2tc outer = add2tc(get_int_type(32), x, m);

  const expr2tc result = outer->simplify();

  // The result must collapse the 2*3 constant fold into a single 6.
  // The exact shape (associativity, operand order) is implementation-defined
  // but no path through the result should still contain both the literal 2
  // and the literal 3 — that would mean reassoc was suppressed.
  REQUIRE(!is_nil_expr(result));
  std::function<bool(const expr2tc &, const BigInt &)> contains_literal =
    [&](const expr2tc &e, const BigInt &lit) -> bool {
    if (is_constant_int2t(e) && to_constant_int2t(e).value == lit)
      return true;
    bool found = false;
    e->foreach_operand([&](const expr2tc &sub) {
      if (!is_nil_expr(sub) && contains_literal(sub, lit))
        found = true;
    });
    return found;
  };
  // After fold the mul subtree should have a 6, and lose either 2 or 3.
  REQUIRE(contains_literal(result, BigInt(6)));
  const bool both =
    contains_literal(result, BigInt(2)) && contains_literal(result, BigInt(3));
  REQUIRE_FALSE(both);
}

TEST_CASE(
  "Reassoc: (x & c1) & c2 folds constants",
  "[bitwise][reassoc][bitand]")
{
  // (x & 0xF0) & 0x0F  -> x & 0  -> 0
  const expr2tc x = symbol2tc(get_uint_type(32), "x");
  const expr2tc c1 = constant_int2tc(get_uint_type(32), BigInt(0xF0));
  const expr2tc c2 = constant_int2tc(get_uint_type(32), BigInt(0x0F));
  const expr2tc inner = bitand2tc(get_uint_type(32), x, c1);
  const expr2tc outer = bitand2tc(get_uint_type(32), inner, c2);

  const expr2tc result = outer->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 0);
}

TEST_CASE("Reassoc: (x | c1) | c2 folds constants", "[bitwise][reassoc][bitor]")
{
  // (x | 0x01) | 0x02  -> x | 0x03
  const expr2tc x = symbol2tc(get_uint_type(32), "x");
  const expr2tc c1 = constant_int2tc(get_uint_type(32), BigInt(0x01));
  const expr2tc c2 = constant_int2tc(get_uint_type(32), BigInt(0x02));
  const expr2tc inner = bitor2tc(get_uint_type(32), x, c1);
  const expr2tc outer = bitor2tc(get_uint_type(32), inner, c2);

  const expr2tc result = outer->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(is_bitor2t(result));
  // The combined constant 0x03 must be present somewhere in the result.
  std::function<bool(const expr2tc &, const BigInt &)> has =
    [&](const expr2tc &e, const BigInt &lit) -> bool {
    if (is_constant_int2t(e) && to_constant_int2t(e).value == lit)
      return true;
    bool found = false;
    e->foreach_operand([&](const expr2tc &s) {
      if (!is_nil_expr(s) && has(s, lit))
        found = true;
    });
    return found;
  };
  REQUIRE(has(result, BigInt(0x03)));
}

TEST_CASE("Reassoc: (x ^ y) ^ x cancels to y", "[bitwise][reassoc][bitxor]")
{
  const expr2tc x = symbol2tc(get_uint_type(32), "x");
  const expr2tc y = symbol2tc(get_uint_type(32), "y");
  const expr2tc inner = bitxor2tc(get_uint_type(32), x, y);
  const expr2tc outer = bitxor2tc(get_uint_type(32), inner, x);

  const expr2tc result = outer->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(result == y);
}

TEST_CASE(
  "Vector mul: scalar zero fold preserves vector type",
  "[arithmetic][mul][vector]")
{
  // mul2t(vec_type, vec, scalar_zero) must NOT return the scalar zero — the
  // result type is vector, so the rewrite needs to broadcast or stay
  // structural. The simplifier's pre-vector scalar shortcuts now skip this
  // case and let distribute_vector_operation handle it.
  const type2tc i32 = get_int_type(32);
  const type2tc vec_type = vector_type2tc(i32, gen_ulong(4));
  std::vector<expr2tc> members{
    constant_int2tc(i32, BigInt(1)),
    constant_int2tc(i32, BigInt(2)),
    constant_int2tc(i32, BigInt(3)),
    constant_int2tc(i32, BigInt(4))};
  const expr2tc vec = constant_vector2tc(vec_type, std::move(members));
  const expr2tc zero = constant_int2tc(i32, BigInt(0));
  const expr2tc m = mul2tc(vec_type, vec, zero);

  const expr2tc result = m->simplify();

  // Either nil (no fold) or vector-typed. The key contract: the result
  // type must NOT be the scalar type of the zero operand.
  if (!is_nil_expr(result))
    REQUIRE(result->type == vec_type);
}

TEST_CASE(
  "Pointer-add fold: same-width sum widens to index_type2",
  "[arithmetic][add][pointer]")
{
  // (p + (s8)100) + (s8)100 — naive same-width fold would wrap to (s8)-56.
  // The simplifier sign-extends both offsets to index_type2 (signed long)
  // and emits the merged constant at that width, preserving the +200
  // semantic that pointer-arith expects after sign-extension.
  const type2tc ptr_type = pointer_type2tc(get_int_type(8));
  const expr2tc p = symbol2tc(ptr_type, "p");
  const expr2tc c1 = constant_int2tc(signedbv_type2tc(8), BigInt(100));
  const expr2tc c2 = constant_int2tc(signedbv_type2tc(8), BigInt(100));
  const expr2tc inner = add2tc(ptr_type, p, c1);
  const expr2tc outer = add2tc(ptr_type, inner, c2);

  const expr2tc result = outer->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(is_add2t(result));
  const add2t &a = to_add2t(result);
  const expr2tc &offset = is_constant_int2t(a.side_2) ? a.side_2 : a.side_1;
  REQUIRE(is_constant_int2t(offset));
  REQUIRE(to_constant_int2t(offset).value == BigInt(200));
  REQUIRE(offset->type == index_type2());
}

TEST_CASE(
  "Pointer-add fold: mixed-width offsets fold to index_type2",
  "[arithmetic][add][pointer]")
{
  // Inner offset signedbv 8 + outer signedbv 32. Both sign-extend to
  // index_type2 before the merge, so the rebuilt constant is 200 typed at
  // signed long — no information loss across the differing widths.
  const type2tc ptr_type = pointer_type2tc(get_int_type(8));
  const expr2tc p = symbol2tc(ptr_type, "p");
  const expr2tc c1 = constant_int2tc(signedbv_type2tc(8), BigInt(100));
  const expr2tc c2 = constant_int2tc(signedbv_type2tc(32), BigInt(100));
  const expr2tc inner = add2tc(ptr_type, p, c1);
  const expr2tc outer = add2tc(ptr_type, inner, c2);

  const expr2tc result = outer->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(is_add2t(result));
  const add2t &a = to_add2t(result);
  const expr2tc &offset = is_constant_int2t(a.side_2) ? a.side_2 : a.side_1;
  REQUIRE(is_constant_int2t(offset));
  REQUIRE(to_constant_int2t(offset).value == BigInt(200));
  REQUIRE(offset->type == index_type2());
}

TEST_CASE(
  "constant_array2t: uniform array does not fold to array_of",
  "[array][simplify]")
{
  // Folding a uniform finite constant_array to constant_array_of is unsound
  // under the SMT encoding: default_convert_array_of writes the initializer
  // at every domain index, including indices past the array's declared size,
  // so OOB reads return the initializer instead of being unconstrained.
  // Pin the contract by checking that constant_array2t::do_simplify leaves
  // a uniform finite array unchanged.
  const type2tc elem_type = get_int_type(32);
  const expr2tc zero = constant_int2tc(elem_type, BigInt(0));
  const expr2tc size = constant_int2tc(get_uint_type(32), BigInt(10));
  const type2tc arr_type = array_type2tc(elem_type, size, false);
  std::vector<expr2tc> members(10, zero);
  const expr2tc arr = constant_array2tc(arr_type, std::move(members));

  // do_simplify is the operator-local rewrite; it must NOT replace this
  // uniform constant_array with an array_of.
  const expr2tc result = arr->do_simplify();
  REQUIRE(is_nil_expr(result));
}

TEST_CASE(
  "Subtraction simplification: p - p uses ptrdiff result type",
  "[arithmetic][sub][pointer]")
{
  // Pointer subtraction has pointer operands but a ptrdiff/integer result
  // type. The self-sub rewrite must use the sub2t's own result type, not
  // the operand type, to avoid synthesizing a pointer-typed zero (NULL)
  // that downstream encoding can't handle.
  const type2tc ptr_type = pointer_type2tc(get_int_type(32));
  const type2tc result_type = signedbv_type2tc(64);
  const expr2tc p = symbol2tc(ptr_type, "p");
  const expr2tc sub = sub2tc(result_type, p, p);

  const expr2tc result = sub->simplify();

  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 0);
  REQUIRE(result->type == result_type);
}

TEST_CASE("Subtraction simplification: x - 0 = x", "[arithmetic][sub]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc zero = constant_int2tc(get_int_type(32), BigInt(0));
  const expr2tc sub = sub2tc(get_int_type(32), x, zero);

  const expr2tc result = sub->simplify();
  REQUIRE(result == x);
}

TEST_CASE("Subtraction constant folding: 10 - 3 = 7", "[arithmetic][sub]")
{
  const expr2tc ten = constant_int2tc(get_int_type(32), BigInt(10));
  const expr2tc three = constant_int2tc(get_int_type(32), BigInt(3));

  const expr2tc sub = sub2tc(get_int_type(32), ten, three);

  const expr2tc result = sub->simplify();

  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 7);
}

TEST_CASE(
  "Subtraction simplification: (base + X) - X = base",
  "[arithmetic][sub]")
{
  const expr2tc base = symbol2tc(get_int_type(32), "base");
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc add = add2tc(get_int_type(32), base, x);

  const expr2tc expr = sub2tc(get_int_type(32), add, x);
  const expr2tc result = expr->simplify();

  REQUIRE(result == base);
}

TEST_CASE("Multiplication simplification: x * 0 = 0", "[arithmetic][mul]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc zero = constant_int2tc(get_int_type(32), BigInt(0));
  const expr2tc mul = mul2tc(get_int_type(32), x, zero);

  const expr2tc result = mul->simplify();

  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 0);
}

TEST_CASE("Multiplication simplification: x * 1 = x", "[arithmetic][mul]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc one = constant_int2tc(get_int_type(32), BigInt(1));
  const expr2tc mul = mul2tc(get_int_type(32), x, one);

  const expr2tc result = mul->simplify();

  REQUIRE(result == x);
}

TEST_CASE("Multiplication simplification: x * (-1) = -x", "[arithmetic][mul]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc minus_one = constant_int2tc(get_int_type(32), BigInt(-1));
  const expr2tc mul = mul2tc(get_int_type(32), x, minus_one);

  const expr2tc result = mul->simplify();

  REQUIRE(is_neg2t(result));
  REQUIRE(to_neg2t(result).value == x);
}

TEST_CASE("Multiplication constant folding: 4 * 5 = 20", "[arithmetic][mul]")
{
  const expr2tc four = constant_int2tc(get_int_type(32), BigInt(4));
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));

  const expr2tc mul = mul2tc(get_int_type(32), four, five);

  const expr2tc result = mul->simplify();

  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 20);
}

TEST_CASE("Division simplification: x / 1 = x", "[arithmetic][div]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc one = constant_int2tc(get_int_type(32), BigInt(1));
  const expr2tc div = div2tc(get_int_type(32), x, one);

  const expr2tc result = div->simplify();

  REQUIRE(result == x);
}

TEST_CASE("Division constant folding: 20 / 4 = 5", "[arithmetic][div]")
{
  const expr2tc twenty = constant_int2tc(get_int_type(32), BigInt(20));
  const expr2tc four = constant_int2tc(get_int_type(32), BigInt(4));
  const expr2tc div = div2tc(get_int_type(32), twenty, four);
  const expr2tc result = div->simplify();

  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 5);
}

TEST_CASE("Modulus simplification: x % x is not folded", "[arithmetic][mod]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc mod = modulus2tc(get_int_type(32), x, x);

  const expr2tc result = mod->simplify();

  REQUIRE(is_nil_expr(result));
}

TEST_CASE(
  "Equality simplification: (x * 3) == 0 -> x == 0 (odd c)",
  "[arithmetic][equal][mul]")
{
  // For odd c, modular bv multiplication is invertible, so (x * c) == 0
  // iff x == 0. Folding is sound.
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc three = constant_int2tc(get_int_type(32), BigInt(3));
  const expr2tc mul = mul2tc(get_int_type(32), x, three);
  const expr2tc zero = constant_int2tc(get_int_type(32), BigInt(0));
  const expr2tc eq = equality2tc(mul, zero);

  const expr2tc result = eq->simplify();

  REQUIRE(!is_nil_expr(result));
  REQUIRE(is_equality2t(result));
  const equality2t &reduced = to_equality2t(result);
  REQUIRE(reduced.side_1 == x);
  REQUIRE(is_constant_int2t(reduced.side_2));
  REQUIRE(to_constant_int2t(reduced.side_2).value == 0);
}

TEST_CASE(
  "Equality simplification: (x * 2) == 0 is not folded (even c)",
  "[arithmetic][equal][mul]")
{
  // For even c, modular bv multiplication is NOT invertible. With c = 2 in
  // 8-bit unsigned, x = 128 satisfies (x * 2) == 0 mod 256 even though
  // x != 0. Folding (x * c) == 0 -> x == 0 would be unsound.
  const expr2tc x = symbol2tc(get_uint_type(8), "x");
  const expr2tc two = constant_int2tc(get_uint_type(8), BigInt(2));
  const expr2tc mul = mul2tc(get_uint_type(8), x, two);
  const expr2tc zero = constant_int2tc(get_uint_type(8), BigInt(0));
  const expr2tc eq = equality2tc(mul, zero);

  const expr2tc result = eq->simplify();

  // Either nil (no rewrite) or unchanged equality structure with the mul
  // intact — the key property is that we do NOT collapse to (x == 0).
  if (!is_nil_expr(result))
  {
    REQUIRE(is_equality2t(result));
    const equality2t &reduced = to_equality2t(result);
    // The mul side must still be a mul (not the bare symbol x).
    REQUIRE_FALSE(reduced.side_1 == x);
    REQUIRE_FALSE(reduced.side_2 == x);
  }
}

TEST_CASE("Modulus simplification: x % 1 = 0", "[arithmetic][mod]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc one = constant_int2tc(get_int_type(32), BigInt(1));
  const expr2tc expr = modulus2tc(get_int_type(32), x, one);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 0);
}

TEST_CASE("Modulus constant folding: 17 % 5 = 2", "[arithmetic][mod]")
{
  const expr2tc seventeen = constant_int2tc(get_int_type(32), BigInt(17));
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc expr = modulus2tc(get_int_type(32), seventeen, five);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 2);
}

TEST_CASE("Logical NOT simplification: !(!x) = x", "[logical][not]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc not_x = not2tc(x);
  const expr2tc expr = not2tc(not_x);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}

TEST_CASE("Logical NOT constant folding: !true = false", "[logical][not]")
{
  const expr2tc true_val = constant_bool2tc(true);
  const expr2tc expr = not2tc(true_val);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == false);
}

TEST_CASE("Logical NOT constant folding: !false = true", "[logical][not]")
{
  const expr2tc false_val = constant_bool2tc(false);
  const expr2tc expr = not2tc(false_val);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE("De Morgan's law: !(x && y) = !x || !y", "[logical][not][demorgan]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc y = symbol2tc(get_bool_type(), "y");
  const expr2tc and_expr = and2tc(x, y);
  const expr2tc expr = not2tc(and_expr);

  const expr2tc result = expr->simplify();

  REQUIRE(is_or2t(result));
  REQUIRE(is_not2t(to_or2t(result).side_1));
  REQUIRE(to_not2t(to_or2t(result).side_1).value == x);
  REQUIRE(is_not2t(to_or2t(result).side_2));
  REQUIRE(to_not2t(to_or2t(result).side_2).value == y);
}

TEST_CASE("De Morgan's law: !(x || y) = !x && !y", "[logical][not][demorgan]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc y = symbol2tc(get_bool_type(), "y");
  const expr2tc or_expr = or2tc(x, y);
  const expr2tc expr = not2tc(or_expr);

  const expr2tc result = expr->simplify();

  REQUIRE(is_and2t(result));
  REQUIRE(is_not2t(to_and2t(result).side_1));
  REQUIRE(to_not2t(to_and2t(result).side_1).value == x);
  REQUIRE(is_not2t(to_and2t(result).side_2));
  REQUIRE(to_not2t(to_and2t(result).side_2).value == y);
}

TEST_CASE("Logical AND simplification: x && x = x", "[logical][and]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc expr = and2tc(x, x);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}

TEST_CASE("Logical AND simplification: x && false = false", "[logical][and]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc false_val = constant_bool2tc(false);
  const expr2tc expr = and2tc(x, false_val);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == false);
}

TEST_CASE("Logical AND simplification: x && true = x", "[logical][and]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc true_val = constant_bool2tc(true);
  const expr2tc expr = and2tc(x, true_val);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}

TEST_CASE("Logical OR simplification: x || x = x", "[logical][or]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc expr = or2tc(x, x);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}

TEST_CASE("Logical OR simplification: x || false = x", "[logical][or]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc false_val = constant_bool2tc(false);
  const expr2tc expr = or2tc(x, false_val);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}

TEST_CASE("Logical OR simplification: x || true = true", "[logical][or]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc true_val = constant_bool2tc(true);
  const expr2tc expr = or2tc(x, true_val);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE("Bitwise AND simplification: x & x = x", "[bitwise][and]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc expr = bitand2tc(get_int_type(32), x, x);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}
TEST_CASE("Bitwise AND simplification: x & 0 = 0", "[bitwise][and]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc zero = constant_int2tc(get_int_type(32), BigInt(0));
  const expr2tc expr = bitand2tc(get_int_type(32), x, zero);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 0);
}

TEST_CASE("Bitwise AND constant folding: 12 & 10 = 8", "[bitwise][and]")
{
  // 00...1100
  const expr2tc twelve = constant_int2tc(get_int_type(32), BigInt(12));
  // 00...1010
  const expr2tc ten = constant_int2tc(get_int_type(32), BigInt(10));
  const expr2tc expr = bitand2tc(get_int_type(32), twelve, ten);

  // 00...1000
  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 8);
}

TEST_CASE("Bitwise OR simplification: x | x = x", "[bitwise][or]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc expr = bitor2tc(get_int_type(32), x, x);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}

TEST_CASE("Bitwise OR simplification: x | 0 = x", "[bitwise][or]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc zero = constant_int2tc(get_int_type(32), BigInt(0));
  const expr2tc expr = bitor2tc(get_int_type(32), x, zero);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}

TEST_CASE("Bitwise OR constant folding: 12 | 10 = 14", "[bitwise][or]")
{
  // 00...1100
  const expr2tc twelve = constant_int2tc(get_int_type(32), BigInt(12));
  // 00...1010
  const expr2tc ten = constant_int2tc(get_int_type(32), BigInt(10));
  const expr2tc expr = bitor2tc(get_int_type(32), twelve, ten);

  // 00...1110
  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 14);
}

TEST_CASE("Bitwise XOR simplification: x ^ x = 0", "[bitwise][xor]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc expr = bitxor2tc(get_int_type(32), x, x);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 0);
}

TEST_CASE("Bitwise XOR simplification: x ^ 0 = x", "[bitwise][xor]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc zero = constant_int2tc(get_int_type(32), BigInt(0));
  const expr2tc expr = bitxor2tc(get_int_type(32), x, zero);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}

TEST_CASE("Bitwise NOT simplification: ~(~x) = x", "[bitwise][not]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc not_x = bitnot2tc(get_int_type(32), x);
  const expr2tc expr = bitnot2tc(get_int_type(32), not_x);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}

TEST_CASE("Shift left simplification: x << 0 = x", "[bitwise][shl]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc zero = constant_int2tc(get_int_type(32), BigInt(0));
  const expr2tc expr = shl2tc(get_int_type(32), x, zero);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}

TEST_CASE("Shift left constant folding: 5 << 2 = 20", "[bitwise][shl]")
{
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc two = constant_int2tc(get_int_type(32), BigInt(2));
  const expr2tc expr = shl2tc(get_int_type(32), five, two);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 20);
}

TEST_CASE(
  "Arithmetic shift right simplification: x >> 0 = x",
  "[bitwise][ashr]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc zero = constant_int2tc(get_int_type(32), BigInt(0));
  const expr2tc expr = ashr2tc(get_int_type(32), x, zero);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}

TEST_CASE("Equality simplification: x == x = true", "[relational][equality]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc expr = equality2tc(x, x);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE("Equality constant folding: 5 == 5 = true", "[relational][equality]")
{
  const expr2tc five_1 = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc five_2 = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc expr = equality2tc(five_1, five_2);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE("Equality constant folding: 5 == 3 = false", "[relational][equality]")
{
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc three = constant_int2tc(get_int_type(32), BigInt(3));
  const expr2tc expr = equality2tc(five, three);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == false);
}

TEST_CASE("Not-equal simplification: x != x = false", "[relational][notequal]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc expr = notequal2tc(x, x);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == false);
}

TEST_CASE("Less-than simplification: x < x = false", "[relational][lessthan]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc expr = lessthan2tc(x, x);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == false);
}

TEST_CASE("Less-than constant folding: 3 < 5 = true", "[relational][lessthan]")
{
  const expr2tc three = constant_int2tc(get_int_type(32), BigInt(3));
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc expr = lessthan2tc(three, five);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE(
  "Greater-than simplification: x > x = false",
  "[relational][greaterthan]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc expr = greaterthan2tc(x, x);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == false);
}

TEST_CASE(
  "Less-than-or-equal simplification: x <= x = true",
  "[relational][lessthanequal]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc expr = lessthanequal2tc(x, x);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE(
  "Greater-than-or-equal simplification: x >= x = true",
  "[relational][greaterthanequal]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc expr = greaterthanequal2tc(x, x);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE(
  "Unsigned comparison: unsigned >= 0 = true",
  "[relational][greaterthanequal]")
{
  const expr2tc x = symbol2tc(get_uint_type(32), "x");
  const expr2tc zero = constant_int2tc(get_uint_type(32), BigInt(0));
  const expr2tc expr = greaterthanequal2tc(x, zero);

  const expr2tc result = expr->simplify();

  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE("If-then-else simplification: cond ? x : x = x", "[conditional][if]")
{
  const expr2tc cond = symbol2tc(get_bool_type(), "cond");
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc expr = if2tc(get_int_type(32), cond, x, x);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}

TEST_CASE("If-then-else simplification: true ? x : y = x", "[conditional][if]")
{
  const expr2tc true_val = constant_bool2tc(true);
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc y = symbol2tc(get_int_type(32), "y");
  const expr2tc expr = if2tc(get_int_type(32), true_val, x, y);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}

TEST_CASE("If-then-else simplification: false ? x : y = y", "[conditional][if]")
{
  const expr2tc false_val = constant_bool2tc(false);
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc y = symbol2tc(get_int_type(32), "y");
  const expr2tc expr = if2tc(get_int_type(32), false_val, x, y);

  const expr2tc result = expr->simplify();

  REQUIRE(result == y);
}

TEST_CASE(
  "Typecast simplification: typecast(x, same_type) = x",
  "[type][typecast]")
{
  type2tc int_type = get_int_type(32);
  const expr2tc x = symbol2tc(int_type, "x");
  const expr2tc expr = typecast2tc(int_type, x);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}

TEST_CASE(
  "Complex simplification: (x + 5) == 10 -> x == 5",
  "[complex][equality]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc ten = constant_int2tc(get_int_type(32), BigInt(10));
  const expr2tc add_expr = add2tc(get_int_type(32), x, five);
  const expr2tc expr = equality2tc(add_expr, ten);

  const expr2tc result = expr->simplify();

  REQUIRE(is_equality2t(result));
  const equality2t &eq = to_equality2t(result);
  REQUIRE(eq.side_1 == x);
  REQUIRE(is_constant_int2t(eq.side_2));
  REQUIRE(to_constant_int2t(eq.side_2).value == 5);
}

TEST_CASE("Absorption law: x && (x || y) = x", "[complex][absorption]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc y = symbol2tc(get_bool_type(), "y");
  const expr2tc or_expr = or2tc(x, y);
  const expr2tc expr = and2tc(x, or_expr);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}

TEST_CASE("Bitwise absorption: x & (x | y) = x", "[complex][absorption]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc y = symbol2tc(get_int_type(32), "y");
  const expr2tc or_expr = bitor2tc(get_int_type(32), x, y);
  const expr2tc expr = bitand2tc(get_int_type(32), x, or_expr);

  const expr2tc result = expr->simplify();

  REQUIRE(result == x);
}

TEST_CASE("Addition simplification: x + ~x = -1", "[arithmetic][add]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc not_x = bitnot2tc(get_int_type(32), x);
  const expr2tc expr = add2tc(get_int_type(32), x, not_x);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == -1);
}

TEST_CASE("Addition simplification: 0 + x = x", "[arithmetic][add]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc zero = constant_int2tc(get_int_type(32), BigInt(0));
  const expr2tc expr = add2tc(get_int_type(32), zero, x);
  const expr2tc result = expr->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(result == x);
}

TEST_CASE("Subtraction simplification: 0 - x = -x", "[arithmetic][sub]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc zero = constant_int2tc(get_int_type(32), BigInt(0));
  const expr2tc expr = sub2tc(get_int_type(32), zero, x);
  const expr2tc result = expr->simplify();
  REQUIRE(is_neg2t(result));
  REQUIRE(to_neg2t(result).value == x);
}

TEST_CASE("Subtraction simplification: x - (-y) = x + y", "[arithmetic][sub]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc y = symbol2tc(get_int_type(32), "y");
  const expr2tc neg_y = neg2tc(get_int_type(32), y);
  const expr2tc expr = sub2tc(get_int_type(32), x, neg_y);
  const expr2tc result = expr->simplify();
  REQUIRE(is_add2t(result));
  REQUIRE(to_add2t(result).side_1 == x);
  REQUIRE(to_add2t(result).side_2 == y);
}

TEST_CASE("Multiplication simplification: 0 * x = 0", "[arithmetic][mul]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc zero = constant_int2tc(get_int_type(32), BigInt(0));
  const expr2tc mul = mul2tc(get_int_type(32), zero, x);
  const expr2tc result = mul->simplify();
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 0);
}

TEST_CASE("Multiplication simplification: 1 * x = x", "[arithmetic][mul]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc one = constant_int2tc(get_int_type(32), BigInt(1));
  const expr2tc mul = mul2tc(get_int_type(32), one, x);
  const expr2tc result = mul->simplify();
  REQUIRE(result == x);
}

TEST_CASE("Division simplification: x / x = 1", "[arithmetic][div]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc div = div2tc(get_int_type(32), x, x);
  const expr2tc result = div->simplify();
  REQUIRE(is_nil_expr(result));
}

TEST_CASE("Negation constant folding: -5 = -5", "[arithmetic][neg]")
{
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc expr = neg2tc(get_int_type(32), five);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == -5);
}

TEST_CASE("Absolute value: abs(5) = 5", "[arithmetic][abs]")
{
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc expr = abs2tc(get_int_type(32), five);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 5);
}

TEST_CASE("Absolute value: abs(-5) = 5", "[arithmetic][abs]")
{
  const expr2tc minus_five = constant_int2tc(get_int_type(32), BigInt(-5));
  const expr2tc expr = abs2tc(get_int_type(32), minus_five);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 5);
}

TEST_CASE("Absolute value: abs(0) = 0", "[arithmetic][abs]")
{
  const expr2tc zero = constant_int2tc(get_int_type(32), BigInt(0));
  const expr2tc expr = abs2tc(get_int_type(32), zero);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 0);
}

TEST_CASE("Logical AND simplification: false && x = false", "[logical][and]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc false_val = constant_bool2tc(false);
  const expr2tc expr = and2tc(false_val, x);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == false);
}

TEST_CASE("Logical AND simplification: true && x = x", "[logical][and]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc true_val = constant_bool2tc(true);
  const expr2tc expr = and2tc(true_val, x);
  const expr2tc result = expr->simplify();
  REQUIRE(result == x);
}

TEST_CASE("Logical OR simplification: true || x = true", "[logical][or]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc true_val = constant_bool2tc(true);
  const expr2tc expr = or2tc(true_val, x);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE("Absorption law: x || (x && y) = x", "[logical][or][absorption]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc y = symbol2tc(get_bool_type(), "y");
  const expr2tc and_expr = and2tc(x, y);
  const expr2tc expr = or2tc(x, and_expr);
  const expr2tc result = expr->simplify();
  REQUIRE(result == x);
}

TEST_CASE("Logical XOR: x ^ false = x", "[logical][xor]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc false_val = constant_bool2tc(false);
  const expr2tc expr = xor2tc(x, false_val);
  const expr2tc result = expr->simplify();
  REQUIRE(result == x);
}

TEST_CASE("Logical XOR constant folding: true ^ false = true", "[logical][xor]")
{
  const expr2tc true_val = constant_bool2tc(true);
  const expr2tc false_val = constant_bool2tc(false);
  const expr2tc expr = xor2tc(true_val, false_val);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE("Logical implies: false => x = true", "[logical][implies]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc false_val = constant_bool2tc(false);
  const expr2tc expr = implies2tc(false_val, x);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE("Logical implies: x => true = true", "[logical][implies]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc true_val = constant_bool2tc(true);
  const expr2tc expr = implies2tc(x, true_val);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE("Bitwise AND simplification: x & ~x = 0", "[bitwise][and]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc not_x = bitnot2tc(get_int_type(32), x);
  const expr2tc expr = bitand2tc(get_int_type(32), x, not_x);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 0);
}

TEST_CASE("Bitwise AND simplification: x & -1 = x", "[bitwise][and]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc minus_one = constant_int2tc(get_int_type(32), BigInt(-1));
  const expr2tc expr = bitand2tc(get_int_type(32), x, minus_one);
  const expr2tc result = expr->simplify();
  REQUIRE(result == x);
}

TEST_CASE("Bitwise AND simplification: -1 & x = x", "[bitwise][and]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc minus_one = constant_int2tc(get_int_type(32), BigInt(-1));
  const expr2tc expr = bitand2tc(get_int_type(32), minus_one, x);
  const expr2tc result = expr->simplify();
  REQUIRE(result == x);
}

TEST_CASE(
  "Bitwise AND consensus: (a | ~b) & (a | b) = a",
  "[bitwise][and][complex]")
{
  const expr2tc a = symbol2tc(get_int_type(32), "a");
  const expr2tc b = symbol2tc(get_int_type(32), "b");
  const expr2tc not_b = bitnot2tc(get_int_type(32), b);
  const expr2tc or1 = bitor2tc(get_int_type(32), a, not_b);
  const expr2tc or2 = bitor2tc(get_int_type(32), a, b);
  const expr2tc expr = bitand2tc(get_int_type(32), or1, or2);
  const expr2tc result = expr->simplify();
  REQUIRE(result == a);
}

TEST_CASE("Bitwise OR simplification: x | ~x = -1", "[bitwise][or]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc not_x = bitnot2tc(get_int_type(32), x);
  const expr2tc expr = bitor2tc(get_int_type(32), x, not_x);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == -1);
}

TEST_CASE("Bitwise OR simplification: x | -1 = -1", "[bitwise][or]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc minus_one = constant_int2tc(get_int_type(32), BigInt(-1));
  const expr2tc expr = bitor2tc(get_int_type(32), x, minus_one);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == -1);
}

TEST_CASE("Bitwise OR simplification: -1 | x = -1", "[bitwise][or]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc minus_one = constant_int2tc(get_int_type(32), BigInt(-1));
  const expr2tc expr = bitor2tc(get_int_type(32), minus_one, x);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == -1);
}

TEST_CASE("Bitwise OR absorption: x | (x & y) = x", "[bitwise][or][absorption]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc y = symbol2tc(get_int_type(32), "y");
  const expr2tc and_expr = bitand2tc(get_int_type(32), x, y);
  const expr2tc expr = bitor2tc(get_int_type(32), x, and_expr);
  const expr2tc result = expr->simplify();
  REQUIRE(result == x);
}

TEST_CASE("Bitwise XOR constant folding: 12 ^ 10 = 6", "[bitwise][xor]")
{
  // 1100 ^ 1010 = 0110 = 6
  const expr2tc twelve = constant_int2tc(get_int_type(32), BigInt(12));
  const expr2tc ten = constant_int2tc(get_int_type(32), BigInt(10));
  const expr2tc expr = bitxor2tc(get_int_type(32), twelve, ten);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 6);
}

TEST_CASE("Bitwise NOT constant folding: ~5 for 8-bit", "[bitwise][not]")
{
  const expr2tc five = constant_int2tc(get_int_type(8), BigInt(5));
  const expr2tc expr = bitnot2tc(get_int_type(8), five);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_int2t(result));
  // ~5 in 8-bit signed = -6 (two's complement)
  REQUIRE(to_constant_int2t(result).value == -6);
}

TEST_CASE(
  "Logical shift right constant folding: 20 >>> 2 = 5",
  "[bitwise][lshr]")
{
  const expr2tc twenty = constant_int2tc(get_uint_type(32), BigInt(20));
  const expr2tc two = constant_int2tc(get_uint_type(32), BigInt(2));
  const expr2tc expr = lshr2tc(get_uint_type(32), twenty, two);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 5);
}

TEST_CASE(
  "Arithmetic shift right constant folding: 20 >> 2 = 5",
  "[bitwise][ashr]")
{
  const expr2tc twenty = constant_int2tc(get_int_type(32), BigInt(20));
  const expr2tc two = constant_int2tc(get_int_type(32), BigInt(2));
  const expr2tc expr = ashr2tc(get_int_type(32), twenty, two);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 5);
}

TEST_CASE("Not-equal constant folding: 5 != 3 = true", "[relational][notequal]")
{
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc three = constant_int2tc(get_int_type(32), BigInt(3));
  const expr2tc expr = notequal2tc(five, three);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE(
  "Not-equal constant folding: 5 != 5 = false",
  "[relational][notequal]")
{
  const expr2tc five_1 = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc five_2 = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc expr = notequal2tc(five_1, five_2);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == false);
}

TEST_CASE("Less-than constant folding: 5 < 3 = false", "[relational][lessthan]")
{
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc three = constant_int2tc(get_int_type(32), BigInt(3));
  const expr2tc expr = lessthan2tc(five, three);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == false);
}

TEST_CASE(
  "Greater-than constant folding: 5 > 3 = true",
  "[relational][greaterthan]")
{
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc three = constant_int2tc(get_int_type(32), BigInt(3));
  const expr2tc expr = greaterthan2tc(five, three);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE(
  "Greater-than constant folding: 3 > 5 = false",
  "[relational][greaterthan]")
{
  const expr2tc three = constant_int2tc(get_int_type(32), BigInt(3));
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc expr = greaterthan2tc(three, five);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == false);
}

TEST_CASE(
  "Less-than-or-equal constant folding: 3 <= 5 = true",
  "[relational][lessthanequal]")
{
  const expr2tc three = constant_int2tc(get_int_type(32), BigInt(3));
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc expr = lessthanequal2tc(three, five);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE(
  "Less-than-or-equal constant folding: 5 <= 3 = false",
  "[relational][lessthanequal]")
{
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc three = constant_int2tc(get_int_type(32), BigInt(3));
  const expr2tc expr = lessthanequal2tc(five, three);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == false);
}

TEST_CASE(
  "Less-than-or-equal constant folding: 5 <= 5 = true",
  "[relational][lessthanequal]")
{
  const expr2tc five_1 = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc five_2 = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc expr = lessthanequal2tc(five_1, five_2);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE(
  "Greater-than-or-equal constant folding: 5 >= 3 = true",
  "[relational][greaterthanequal]")
{
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc three = constant_int2tc(get_int_type(32), BigInt(3));
  const expr2tc expr = greaterthanequal2tc(five, three);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE(
  "Greater-than-or-equal constant folding: 3 >= 5 = false",
  "[relational][greaterthanequal]")
{
  const expr2tc three = constant_int2tc(get_int_type(32), BigInt(3));
  const expr2tc five = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc expr = greaterthanequal2tc(three, five);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == false);
}

TEST_CASE(
  "Greater-than-or-equal constant folding: 5 >= 5 = true",
  "[relational][greaterthanequal]")
{
  const expr2tc five_1 = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc five_2 = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc expr = greaterthanequal2tc(five_1, five_2);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE(
  "If-then-else simplification: cond ? true : false = cond",
  "[conditional][if]")
{
  const expr2tc cond = symbol2tc(get_bool_type(), "cond");
  const expr2tc true_val = constant_bool2tc(true);
  const expr2tc false_val = constant_bool2tc(false);
  const expr2tc expr = if2tc(get_bool_type(), cond, true_val, false_val);
  const expr2tc result = expr->simplify();
  REQUIRE(result == cond);
}

TEST_CASE(
  "If-then-else simplification: cond ? false : true = !cond",
  "[conditional][if]")
{
  const expr2tc cond = symbol2tc(get_bool_type(), "cond");
  const expr2tc true_val = constant_bool2tc(true);
  const expr2tc false_val = constant_bool2tc(false);
  const expr2tc expr = if2tc(get_bool_type(), cond, false_val, true_val);
  const expr2tc result = expr->simplify();
  REQUIRE(is_not2t(result));
  REQUIRE(to_not2t(result).value == cond);
}

TEST_CASE("object_size regression", "[pointer_object]")
{
  // x: char[50]
  // OFFSET_OF(&x[0] + 10) = 10

  const type2tc arr_t = array_type2tc(
    get_int_type(8), constant_int2tc(get_uint_type(64), BigInt(50)), false);

  const expr2tc x = symbol2tc(arr_t, "x");
  const expr2tc zero_index =
    index2tc(get_int_type(8), x, gen_zero(get_int_type(64)));

  const expr2tc side_1 = address_of2tc(get_int_type(8), zero_index);

  const expr2tc side_2 = constant_int2tc(get_int_type(32), BigInt(10));

  config.ansi_c.address_width = 64;
  const expr2tc pointer_offset = pointer_offset2tc(
    get_int_type(config.ansi_c.address_width),
    add2tc(pointer_type2tc(get_int_type(8)), side_1, side_2));

  const expr2tc result = pointer_offset->simplify();

  REQUIRE(!is_nil_expr(result));
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 10);
}

TEST_CASE(
  "byte_extract(byte_update(src, OFF, v), OFF) folds when in bounds",
  "[byte][simplify]")
{
  // For an in-bounds offset the read-after-write fold is sound: SMT
  // byte_update writes the byte at OFF, then byte_extract at the same OFF
  // reads it back as v.
  const type2tc u32 = get_uint_type(32);
  const type2tc u8 = get_uint_type(8);
  const expr2tc src = symbol2tc(u32, "src");
  const expr2tc v = symbol2tc(u8, "v");
  const expr2tc off = constant_int2tc(get_int_type(32), BigInt(1));
  const expr2tc bu = byte_update2tc(u32, src, off, v, false);
  const expr2tc be = byte_extract2tc(u8, bu, off, false);

  const expr2tc result = be->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(result == v);
}

TEST_CASE(
  "Bitwise reassoc: skip fold for >64-bit destination types",
  "[bitwise][reassoc][wide]")
{
  // bitwise_fold uses a 64-bit round-trip, which can't preserve
  // sign-extension above bit 63 for signedbv > 64. (s128)-2 & (s128)-4
  // would fold via uint64 to 2^64-4 and from_integer would zero-extend
  // to the destination — wrong, since the correct result is -4 in s128.
  // Refuse the fold entirely for >64-bit destination types.
  const type2tc s128 = signedbv_type2tc(128);
  const expr2tc x = symbol2tc(s128, "x");
  const expr2tc neg2 = constant_int2tc(s128, BigInt(-2));
  const expr2tc neg4 = constant_int2tc(s128, BigInt(-4));
  // Build a chain (x & -2) & -4 — reassoc would try to fold the
  // constants. With the >64-bit guard, the fold is refused.
  const expr2tc inner = bitand2tc(s128, x, neg2);
  const expr2tc outer = bitand2tc(s128, inner, neg4);

  const expr2tc result = outer->simplify();

  // The result must NOT contain a positive 2^64-4 constant — that would
  // be the buggy zero-extended fold.
  std::function<bool(const expr2tc &)> contains_buggy =
    [&](const expr2tc &e) -> bool {
    if (is_constant_int2t(e))
    {
      const BigInt &v = to_constant_int2t(e).value;
      // 2^64 - 4 = 18446744073709551612.
      return v == BigInt::power2(64) - 4;
    }
    bool found = false;
    e->foreach_operand([&](const expr2tc &s) {
      if (!is_nil_expr(s) && contains_buggy(s))
        found = true;
    });
    return found;
  };
  if (!is_nil_expr(result))
    REQUIRE_FALSE(contains_buggy(result));
}

TEST_CASE(
  "Short-circuit pre-pass: if(true, x, dead) folds without walking dead arm",
  "[short-circuit][simplify]")
{
  // Pin contract: when the top-level expression is and/or/if, the
  // simplifier tries do_simplify() before the operand walk. For
  // if(true, x, dead) that returns x directly, and the dead arm is
  // never simplified — useful both for cheap short-circuit and to
  // avoid dyn_sized_array_excp on a dead arm aborting the whole fold.
  const type2tc i32 = get_int_type(32);
  const expr2tc x = symbol2tc(i32, "x");
  const expr2tc dead = symbol2tc(i32, "dead");
  const expr2tc cond = gen_true_expr();
  const expr2tc sel = if2tc(i32, cond, x, dead);

  const expr2tc result = sel->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(result == x);
}

TEST_CASE(
  "Short-circuit pre-pass: false && X folds to false",
  "[short-circuit][simplify]")
{
  const type2tc bool_t = get_bool_type();
  const expr2tc f = gen_false_expr();
  const expr2tc x = symbol2tc(bool_t, "x");
  const expr2tc a = and2tc(f, x);
  const expr2tc result = a->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == false);
}

TEST_CASE(
  "Short-circuit pre-pass: true || X folds to true",
  "[short-circuit][simplify]")
{
  const type2tc bool_t = get_bool_type();
  const expr2tc t = gen_true_expr();
  const expr2tc x = symbol2tc(bool_t, "x");
  const expr2tc o = or2tc(t, x);
  const expr2tc result = o->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == true);
}

TEST_CASE(
  "Reassoc: nested chain inside same-chain child still canonicalizes",
  "[arithmetic][reassoc][nested]")
{
  // x + (y + ((a*2)*(b*3))) — the inner add is a same-chain child of
  // the outer add, but the mul subtree under it is a *different* chain
  // kind. An earlier version of operand-walk suppression OR'd a
  // "same-chain" bit into the operand's suppress_reassoc, which then
  // propagated through to the mul grandchild and disabled its reassoc.
  // After the fix, the mul should still canonicalize into 6*a*b.
  const type2tc i32 = get_int_type(32);
  const expr2tc x = symbol2tc(i32, "x");
  const expr2tc y = symbol2tc(i32, "y");
  const expr2tc a = symbol2tc(i32, "a");
  const expr2tc b = symbol2tc(i32, "b");
  const expr2tc two = constant_int2tc(i32, BigInt(2));
  const expr2tc three = constant_int2tc(i32, BigInt(3));
  const expr2tc a2 = mul2tc(i32, a, two);
  const expr2tc b3 = mul2tc(i32, b, three);
  const expr2tc m = mul2tc(i32, a2, b3);
  const expr2tc inner = add2tc(i32, y, m);
  const expr2tc outer = add2tc(i32, x, inner);

  const expr2tc result = outer->simplify();
  REQUIRE(!is_nil_expr(result));

  // After fold the mul subtree must contain a 6, with neither a 2 nor
  // a 3 surviving (those would mean reassoc was suppressed).
  std::function<bool(const expr2tc &, const BigInt &)> contains_literal =
    [&](const expr2tc &e, const BigInt &lit) -> bool {
    if (is_constant_int2t(e) && to_constant_int2t(e).value == lit)
      return true;
    bool found = false;
    e->foreach_operand([&](const expr2tc &s) {
      if (!is_nil_expr(s) && contains_literal(s, lit))
        found = true;
    });
    return found;
  };
  REQUIRE(contains_literal(result, BigInt(6)));
  const bool both =
    contains_literal(result, BigInt(2)) && contains_literal(result, BigInt(3));
  REQUIRE_FALSE(both);
}

TEST_CASE(
  "Concat: contiguous LE byte_extracts collapse to source",
  "[concat][simplify]")
{
  // concat(byte_extract(x:u32, 3), byte_extract(x:u32, 2),
  //        byte_extract(x:u32, 1), byte_extract(x:u32, 0))
  // is the little-endian reconstruction of x. After simplification it
  // should fold back to x. The collect_leaves change must not break
  // this fold — the lambda walks side_1/side_2 directly instead of
  // cloning the outer concat.
  const type2tc u32 = unsignedbv_type2tc(32);
  const type2tc u8 = unsignedbv_type2tc(8);
  const type2tc i32 = signedbv_type2tc(32);
  const expr2tc x = symbol2tc(u32, "x");
  auto bx = [&](unsigned off) {
    return byte_extract2tc(
      u8, x, constant_int2tc(i32, BigInt(off)), false /* little endian */);
  };
  // Build concat(bx(3), concat(bx(2), concat(bx(1), bx(0))))
  const expr2tc inner = concat2tc(unsignedbv_type2tc(16), bx(1), bx(0));
  const expr2tc mid = concat2tc(unsignedbv_type2tc(24), bx(2), inner);
  const expr2tc outer = concat2tc(u32, bx(3), mid);

  const expr2tc result = outer->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(result == x);
}

TEST_CASE(
  "Shift combine: refuse fold when sum doesn't fit shift-count type",
  "[shift][simplify]")
{
  // (u128 x << u5(20)) << u5(20). The combined count (40) doesn't fit
  // in u5 (range 0..31), so building from_integer(40, u5) would wrap to
  // 8 and produce x << 8 instead of x << 40. Refuse the fold.
  const type2tc u128 = unsignedbv_type2tc(128);
  const type2tc u5 = unsignedbv_type2tc(5);
  const expr2tc x = symbol2tc(u128, "x");
  const expr2tc twenty = constant_int2tc(u5, BigInt(20));
  const expr2tc inner = shl2tc(u128, x, twenty);
  const expr2tc outer = shl2tc(u128, inner, twenty);

  const expr2tc result = outer->simplify();

  // Either nil (no fold) or a structurally unchanged outer shl. The key
  // contract: the result must NOT be a single x << 8.
  if (!is_nil_expr(result))
  {
    REQUIRE(is_shl2t(result));
    const shl2t &s = to_shl2t(result);
    // If folded, the shift amount must be the correct 40 in some wider
    // representation — never the wrapped value 8 in u5.
    if (is_constant_int2t(s.side_2))
    {
      const BigInt &amt = to_constant_int2t(s.side_2).value;
      REQUIRE_FALSE(amt == BigInt(8));
    }
  }
}

TEST_CASE("popcount: signed -1 has all bits set", "[popcount][simplify]")
{
  // -1 in signedbv 32 has 32 one-bits in two's complement. The previous
  // implementation counted '1' chars in integer2string(value, 2), which
  // produced "-1" for negative values and reported just one '1'. Use
  // integer2binary at the operand width.
  const type2tc s32 = signedbv_type2tc(32);
  const expr2tc minus_one = constant_int2tc(s32, BigInt(-1));
  const expr2tc pc = popcount2tc(minus_one);
  const expr2tc result = pc->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 32);
}

TEST_CASE("bswap: u32 0x12345678 -> 0x78563412", "[bswap][simplify]")
{
  // Sanity check that the basic bswap fold still works after the
  // negative-value normalization fix.
  const type2tc u32 = get_uint_type(32);
  const expr2tc v = constant_int2tc(u32, BigInt(0x12345678));
  const expr2tc bs = bswap2tc(u32, v);
  const expr2tc result = bs->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == BigInt(0x78563412));
}

TEST_CASE("bswap: signedbv 32 -1 stays all-ones", "[bswap][simplify]")
{
  // -1 is 0xFFFFFFFF in two's complement; byte-reversing all-ones
  // yields all-ones. The previous fold computed `(v >> bit) % 256`
  // on a signed BigInt -1, which gives signed-magnitude byte values
  // that don't combine back into 0xFFFFFFFF. After normalizing to the
  // unsigned bit pattern, the result must still be -1 in signed form.
  const type2tc s32 = signedbv_type2tc(32);
  const expr2tc v = constant_int2tc(s32, BigInt(-1));
  const expr2tc bs = bswap2tc(s32, v);
  const expr2tc result = bs->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == BigInt(-1));
}

TEST_CASE(
  "byte_update with huge constant offset returns nil instead of throwing",
  "[byte][simplify]")
{
  // Pin overflow safety: a constant byte_update offset near UINT64_MAX/8
  // must NOT trigger string::replace at a position past end. The
  // simplifier should bail out (return nil) rather than throw.
  const type2tc u32 = get_uint_type(32);
  const type2tc u8 = get_uint_type(8);
  // Source: 4-byte constant 0
  const expr2tc src = constant_int2tc(u32, BigInt(0));
  // Update value: a constant byte
  const expr2tc v = constant_int2tc(u8, BigInt(0xAB));
  // Offset: huge, within uint64 but src_offset * 8 must overflow size
  // checks once added to value.length()
  const BigInt huge_offset = (BigInt(1) << 60);
  const expr2tc off = constant_int2tc(get_uint_type(64), huge_offset);
  const expr2tc bu = byte_update2tc(u32, src, off, v, false);

  // Must not throw.
  const expr2tc result = bu->simplify();
  // Either nil (refused) or a structurally-unchanged byte_update; the
  // contract is: no exception, and the simplifier doesn't fabricate a
  // value at the impossible position.
  if (!is_nil_expr(result))
    REQUIRE(is_byte_update2t(result));
}

TEST_CASE(
  "byte_extract(byte_update(src, OOB, v), OOB) is not folded",
  "[byte][simplify]")
{
  // For an out-of-bounds offset the SMT byte_update is a no-op (returns
  // src unchanged) and byte_extract returns the OOB sentinel. Folding to
  // the update value would replace those backend semantics with the
  // would-be-written byte, hiding any OOB-read effect.
  const type2tc u32 = get_uint_type(32);
  const type2tc u8 = get_uint_type(8);
  const expr2tc src = symbol2tc(u32, "src");
  const expr2tc v = symbol2tc(u8, "v");
  // u32 has 4 bytes; offset 5 is OOB.
  const expr2tc off_oob = constant_int2tc(get_int_type(32), BigInt(5));
  const expr2tc bu = byte_update2tc(u32, src, off_oob, v, false);
  const expr2tc be = byte_extract2tc(u8, bu, off_oob, false);

  const expr2tc result = be->simplify();
  // Either nil (no fold) or unchanged shape — the key property is that
  // we do NOT collapse to v.
  if (!is_nil_expr(result))
    REQUIRE_FALSE(result == v);
}

// TODO: Tests that should be valid but... not yet!

#if 0
TEST_CASE("Division simplification: 0 / x = 0", "[arithmetic][div]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc zero = constant_int2tc(get_int_type(32), BigInt(0));
  const expr2tc div = div2tc(get_int_type(32), zero, x);
  const expr2tc result = div->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 0);
}

TEST_CASE("Modulus simplification: 0 % x = 0", "[arithmetic][mod]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc zero = constant_int2tc(get_int_type(32), BigInt(0));
  const expr2tc expr = modulus2tc(get_int_type(32), zero, x);
  const expr2tc result = expr->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 0);
}

TEST_CASE("Negation simplification: -(-x) = x", "[arithmetic][neg]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc neg_x = neg2tc(get_int_type(32), x);
  const expr2tc expr = neg2tc(get_int_type(32), neg_x);
  const expr2tc result = expr->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(result == x);
}

TEST_CASE("Logical XOR: x ^ x = false", "[logical][xor]")
{
  const expr2tc x = symbol2tc(get_bool_type(), "x");
  const expr2tc expr = xor2tc(x, x);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == false);
}
TEST_CASE("Logical implies: true => false = false", "[logical][implies]")
{
  const expr2tc true_val = constant_bool2tc(true);
  const expr2tc false_val = constant_bool2tc(false);
  const expr2tc expr = implies2tc(true_val, false_val);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_bool2t(result));
  REQUIRE(to_constant_bool2t(result).value == false);
}
TEST_CASE("Bitwise XOR simplification: x ^ ~x = -1", "[bitwise][xor]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc not_x = bitnot2tc(get_int_type(32), x);
  const expr2tc expr = bitxor2tc(get_int_type(32), x, not_x);
  const expr2tc result = expr->simplify();
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == -1);
}
TEST_CASE("Bitwise XOR simplification: x ^ -1 = ~x", "[bitwise][xor]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc minus_one = constant_int2tc(get_int_type(32), BigInt(-1));
  const expr2tc expr = bitxor2tc(get_int_type(32), x, minus_one);
  const expr2tc result = expr->simplify();
  REQUIRE(is_bitnot2t(result));
  REQUIRE(to_bitnot2t(result).value == x);
}
TEST_CASE("Logical shift right: x >>> 0 = x", "[bitwise][lshr]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc zero = constant_int2tc(get_int_type(32), BigInt(0));
  const expr2tc expr = lshr2tc(get_int_type(32), x, zero);
  const expr2tc result = expr->simplify();
  REQUIRE(result == x);
}
#endif
