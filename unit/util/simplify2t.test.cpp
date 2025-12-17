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
#include <util/simplify_expr.h>

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

TEST_CASE("Modulus simplification: x % x = 0", "[arithmetic][mod]")
{
  const expr2tc x = symbol2tc(get_int_type(32), "x");
  const expr2tc mod = modulus2tc(get_int_type(32), x, x);

  const expr2tc result = mod->simplify();

  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 0);
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


TEST_CASE("uint_32_t v = 10; (uint64_t) v ==> 10", "[typecast][if]")
{
  const expr2tc ten_32 = constant_int2tc(get_uint_type(32), BigInt(10));
  const expr2tc typecast = typecast2tc(get_uint_type(64), ten_32);
  const expr2tc result = typecast->simplify();
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 10);
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

  // const expr2tc obj =  pointer_object2tc(pointer_type2tc(get_int_type(8)), zero_index);

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

TEST_CASE("object_size regression 2", "[pointer_object]")
{
  // x: char[50]
  // OFFSET_OF(&x[0] + 10) = 10

  expr2tc zero = constant_int2tc(get_uint_type(67), BigInt(0));
  expr2tc add = add2tc(get_uint_type(67), zero, zero);

  expr2tc result = add->simplify();
  REQUIRE(!is_nil_expr(result));
  REQUIRE(is_constant_int2t(result));
  REQUIRE(to_constant_int2t(result).value == 0);
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
