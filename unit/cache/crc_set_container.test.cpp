/*******************************************************************\
 Module: CRC Set Container unit tests
 Author: Rafael Sá Menezes
 Date: April 2020
 Test Plan:
 - Check if trivial cases are ok
 - Check if
 - Multiple operators (not working)
\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <cache/cache.h>
#include "cache_test_utils.h"

// ******************** TESTS ********************

TEST_CASE("crc set container", "[caching]")
{
  SECTION("expression_true_saved_ok")
  {
    init_test_values();
    ssa_set_container::clear_cache();
    expr2tc expr = equality_1_ordered();
    ssa_set_container storage;

    crc_expr items;
    items.insert(expr->crc());
    storage.add(items);
    REQUIRE(storage.check(items));
  }

  SECTION("expression_default_ok")
  {
    init_test_values();
    ssa_set_container::clear_cache();
    expr2tc expr = equality_1_ordered();
    ssa_set_container storage;
    crc_expr items;
    items.insert(expr->crc());
    REQUIRE(!storage.check(items));
  }

  SECTION("expression_multiple_false_true_ok")
  {
    init_test_values();
    ssa_set_container::clear_cache();
    expr2tc expr1 = equality_1();
    expr2tc expr2 = equality_2();
    ssa_set_container storage;

    crc_expr items_1;
    items_1.insert(expr1->crc());

    crc_expr items_2;
    items_2.insert(expr2->crc());

    storage.add(items_1);

    REQUIRE(storage.check(items_1));
    REQUIRE(!storage.check(items_2));
  }

  SECTION("expression_manually_built_ok")
  {
    init_test_values();
    ssa_set_container::clear_cache();
    // ((x + y) + -2) == 0
    expr2tc expr1 = equality_1_green_normal();

    symbol2tc Y = create_unsigned_32_symbol_expr("Y");
    symbol2tc X = create_unsigned_32_symbol_expr("X");
    constant_int2tc minus_two = create_signed_32_value_expr(-2);
    constant_int2tc zero = create_signed_32_value_expr(0);
    add2tc add_1 = create_signed_32_add_expr(X, Y);
    add2tc add_2 = create_signed_32_add_expr(add_1, minus_two);
    expr2tc expr2 = create_equality_relation(add_2, zero);

    crc_expr items_1;
    items_1.insert(expr1->crc());

    crc_expr items_2;
    items_2.insert(expr2->crc());

    ssa_set_container storage;
    storage.add(items_1);
    storage.add(items_2);

    REQUIRE(storage.check(items_1));
    REQUIRE(storage.check(items_2));
  }

  SECTION("expression_manually_built_inner_expr")
  {
    /*
   * This is a big test, it will:
   *
   * 1. Add (X < 0) AND (X>0) AS UNSAT
   * 2. Check whether (X < 0) AND (X>0) is UNSAT
   * 3. Check whether X < 0 is unknown
   * 4. Check whether X > 0 is unknown
   * 5. Check whether (X < 0) AND (X>0) AND (X == 0) is UNSAT
  */
    ssa_set_container::clear_cache();
    symbol2tc X = create_unsigned_32_symbol_expr("X");
    constant_int2tc zero = create_signed_32_value_expr(0);

    // expr1 -> X < 0
    expr2tc expr1 = create_lesser_relation(X, zero);
    // expr2 -> X > 0
    expr2tc expr2 = create_greater_relation(X, zero);
    // expr3 -> X == 0
    expr2tc expr3 = create_equality_relation(X, zero);

    // X < 0 AND X > 0
    crc_expr items_1;
    items_1.insert(expr1->crc());
    items_1.insert(expr2->crc());
    // X < 0
    crc_expr items_2;
    items_2.insert(expr1->crc());
    // X > 0
    crc_expr items_3;
    items_3.insert(expr2->crc());
    // X < 0 AND X > 0 AND X == 0
    crc_expr items_4;
    items_4.insert(expr1->crc());
    items_4.insert(expr2->crc());
    items_4.insert(expr3->crc());

    ssa_set_container storage;
    // 1. Add (X < 0) AND (X>0) AS UNSAT
    storage.add(items_1);

    // 2. Check if (X < 0) AND (X>0) is UNSAT
    REQUIRE(storage.check(items_1));
    // 3. Check if X < 0 is unknown
    REQUIRE(!storage.check(items_2));
    // 4. Check if X > 0 is unknown
    REQUIRE(!storage.check(items_3));
    // 5. Check if (X < 0) AND (X>0) AND (X == 0) is UNSAT
    REQUIRE(storage.check(items_4));
  }
}
