#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>

#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/namespace.h>
#include <util/simplify_expr.h>
#include <util/std_expr.h>
#include <util/std_types.h>

TEST_CASE("for a division expression...", "[unit][util][std_expr]")
{
  typet t("signedbv");
  auto dividend = from_integer(10, t);
  auto divisor = from_integer(5, t);
  auto div = div_exprt(dividend, divisor);

  SECTION("its divisor and dividend have the values assigned to them")
  {
    REQUIRE(div.op0() == dividend);
    REQUIRE(div.op1() == divisor);
  }
  /* This will not work on current implementation (and we probably shouldn't waste time on this)
   * after irep is switched to irep2 we have to fix this
  SECTION("its type is that of its operands")
  {
    REQUIRE(div.type() == t);
  }
  */
}