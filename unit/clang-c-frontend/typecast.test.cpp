/*******************************************************************
 Module: Typecast unit test

 Author: Rafael Sá Menezes

 Date: January 2021

 Test Plan:
   - ToUnion (builtins, exceptions)
 \*******************************************************************/

#include "../testing-utils/util_irep.h"

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <clang-c-frontend/clang_c_convert.h>
#include <clang-c-frontend/typecast.h>
#include <util/type.h>
#include <util/expr_util.h>

// ******************** TESTS ********************

namespace
{
void gen_typecast_to_union(exprt &dest, const typet &type)
{
  clang_c_convertert::gen_typecast_to_union(dest, type);
}
} // namespace

SCENARIO("ToUnion typecast construction", "[core][clang-c-frontend][typecast]")
{
  GIVEN("Some components")
  {
    auto component1 = gen_component("var_1", Builtin_Type::Int);
    auto component2 = gen_component("var_2", Builtin_Type::UInt);
    auto component3 = gen_component("var_3", Builtin_Type::UInt);
    auto component4 = gen_component("var_4", Builtin_Type::Double);

    union_typet t;
    typet builtin;

    AND_GIVEN("An union_type {int;}")
    {
      t.components().push_back(component1);
      THEN("t should contain a Int")
      {
        gen_builtin_type(builtin, Builtin_Type::Int);
        exprt e = gen_zero(builtin);
        CHECK_NOTHROW(gen_typecast_to_union(e, t));
        CHECK(to_union_expr(e).get_component_name() == "var_1");
        CHECK(to_union_expr(e).op0().type() == component1.type());
      }
    }

    AND_GIVEN("An union type {int;uint;}")
    {
      t.components().push_back(component1);
      t.components().push_back(component2);

      THEN("t should contain a UInt")
      {
        gen_builtin_type(builtin, Builtin_Type::UInt);
        exprt e = gen_zero(builtin);
        CHECK_NOTHROW(gen_typecast_to_union(e, t));
        CHECK(to_union_expr(e).get_component_name() == "var_2");
        CHECK(to_union_expr(e).op0().type() == component2.type());
      }
    }

    AND_GIVEN("An union type {double;int;}")
    {
      t.components().push_back(component4);
      t.components().push_back(component1);

      THEN("t should contain a Int")
      {
        gen_builtin_type(builtin, Builtin_Type::Int);
        exprt e = gen_zero(builtin);
        CHECK_NOTHROW(gen_typecast_to_union(e, t));
        CHECK(to_union_expr(e).get_component_name() == "var_1");
        CHECK(to_union_expr(e).op0().type() == component1.type());
      }
    }

    AND_GIVEN("An union type {double;uint;uint}")
    {
      t.components().push_back(component4);
      t.components().push_back(component2);
      t.components().push_back(component3);

      THEN("t should contain Uint and return component2")
      {
        gen_builtin_type(builtin, Builtin_Type::UInt);
        exprt e = gen_zero(builtin);
        CHECK_NOTHROW(gen_typecast_to_union(e, t));
        CHECK(to_union_expr(e).get_component_name() == "var_2");
        CHECK(to_union_expr(e).op0().type() == component2.type());
      }
    }
  }
}
