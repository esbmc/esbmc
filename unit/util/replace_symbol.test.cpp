/*******************************************************************\

Module: replace_symbolt unit tests

Author: Michael Tautschnig

\*******************************************************************/

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>

#include <util/replace_symbol.h>
#include <util/std_expr.h>
#include <util/std_types.h>

TEST_CASE("Replace all symbols in expression", "[core][util][replace_symbol]")
{
  symbol_exprt s1("a", typet("some_type"));
  symbol_exprt s2("b", typet("some_type"));

  binary_exprt binary(s1, "binary", s2, typet("some_type"));

  array_typet array_type(typet("sub-type"), s1);
  REQUIRE(array_type.size() == s1);

  exprt other_expr("other", typet("some_type"));

  replace_symbolt r;
  r.insert(s1.identifier(), other_expr);

  REQUIRE(!r.replace(binary));
  REQUIRE(binary.op0() == other_expr);

  REQUIRE(!r.replace(s1));
  REQUIRE(s1 == other_expr);

  REQUIRE(r.replace(s2));
  REQUIRE(s2 == symbol_exprt("b", typet("some_type")));

  // REQUIRE(!r.replace(array_type)); TODO: Fix this
  // REQUIRE(array_type.size() == other_expr);
}

TEST_CASE("Lvalue only", "[core][util][replace_symbol]")
{
  symbol_exprt s1("a", typet("some_type"));
  array_typet array_type(typet("some_type"), s1);
  symbol_exprt array("b", array_type);
  index_exprt index(array, s1);

  binary_exprt binary(
    address_of_exprt(s1),
    "binary",
    address_of_exprt(index),
    typet("some_type"));

  constant_exprt c("some_value", "some_value", typet("some_type"));

  replace_symbolt r;
  r.insert(s1.identifier(), c);

  REQUIRE(!r.replace(binary));
  // REQUIRE(binary.op0() == address_of_exprt(s1));
  const index_exprt &index_expr =
    to_index_expr(to_address_of_expr(binary.op1()).object());
  //REQUIRE(to_array_type(index_expr.array().type()).size() == c);
  REQUIRE(index_expr.index() == c);

  address_of_exprt address_of(s1);
  // r.erase("a");
  r.insert(s1.identifier(), address_of);

  REQUIRE(r.replace(binary));
  // REQUIRE(binary.op0() == address_of_exprt(s1));
}

TEST_CASE("Replace always", "[core][util][replace_symbol]")
{
  symbol_exprt s1("a", typet("some_type"));
  array_typet array_type(typet("some_type"), s1);
  symbol_exprt array("b", array_type);
  index_exprt index(array, s1);

  binary_exprt binary(
    address_of_exprt(s1),
    "binary",
    address_of_exprt(index),
    typet("some_type"));

  constant_exprt c("some_value", "some_value", typet("some_type"));

  replace_symbolt r;
  r.insert(s1.identifier(), c);

  REQUIRE(!r.replace(binary));
  REQUIRE(binary.op0() == address_of_exprt(c));
  const index_exprt &index_expr =
    to_index_expr(to_address_of_expr(binary.op1()).object());
  //REQUIRE(to_array_type(index_expr.array().type()).size() == c);
  REQUIRE(index_expr.index() == c);
}
