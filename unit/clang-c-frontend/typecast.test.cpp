/*******************************************************************
 Module: Typecast unit test

 Author: Rafael Sá Menezes

 Date: January 2021

 Test Plan:
   - ToUnion (builtins, exceptions)
 \*******************************************************************/

#define BOOST_TEST_MODULE "Typecast Test"
#include <clang-c-frontend/typecast.h>
#include <util/type.h>
#include <util/expr_util.h>
#include "util.h"
#include <boost/test/included/unit_test.hpp>
namespace utf = boost::unit_test;

// ******************** TESTS ********************

BOOST_AUTO_TEST_SUITE(ToUnion)
BOOST_AUTO_TEST_CASE(basic_union_1)
{
  auto component = gen_component("var_1", Builtin_Type::Int);

  union_typet t;
  t.components().push_back(component);

  typet builtin;
  gen_builtin_type(builtin, Builtin_Type::Int);
  exprt e = gen_zero(builtin);

  gen_typecast_to_union(e, t);

  BOOST_TEST(to_union_expr(e).get_component_name() == "var_1");
  BOOST_TEST(to_union_expr(e).op0().type() == component.type());
}

BOOST_AUTO_TEST_CASE(basic_union_1_error)
{
  auto component = gen_component("var_1", Builtin_Type::Int);

  union_typet t;
  t.components().push_back(component);

  typet builtin;
  gen_builtin_type(builtin, Builtin_Type::UInt);
  exprt e = gen_zero(builtin);

  BOOST_CHECK_THROW(gen_typecast_to_union(e, t), std::domain_error);
}

BOOST_AUTO_TEST_CASE(basic_union_2)
{
  auto component = gen_component("var_1", Builtin_Type::Int);
  auto component2 = gen_component("var_2", Builtin_Type::UInt);

  union_typet t;
  t.components().push_back(component);
  t.components().push_back(component2);

  typet builtin;
  gen_builtin_type(builtin, Builtin_Type::UInt);
  exprt e = gen_zero(builtin);

  BOOST_CHECK_NO_THROW(gen_typecast_to_union(e, t));
  BOOST_TEST(to_union_expr(e).get_component_name() == "var_2");
  BOOST_TEST(to_union_expr(e).op0().type() == component2.type());
}

BOOST_AUTO_TEST_CASE(basic_union_3)
{
  auto component = gen_component("var_1", Builtin_Type::Double);
  auto component2 = gen_component("var_2", Builtin_Type::UInt);

  union_typet t;
  t.components().push_back(component);
  t.components().push_back(component2);

  typet builtin;
  gen_builtin_type(builtin, Builtin_Type::UInt);
  exprt e = gen_zero(builtin);

  BOOST_CHECK_NO_THROW(gen_typecast_to_union(e, t));
  BOOST_TEST(to_union_expr(e).get_component_name() == "var_2");
  BOOST_TEST(to_union_expr(e).op0().type() == component2.type());
}

BOOST_AUTO_TEST_CASE(basic_union_4)
{
  auto component = gen_component("var_1", Builtin_Type::Double);
  auto component2 = gen_component("var_2", Builtin_Type::UInt);
  auto component3 = gen_component("var_3", Builtin_Type::UInt);

  union_typet t;
  t.components().push_back(component);
  t.components().push_back(component2);
  t.components().push_back(component3);

  typet builtin;
  gen_builtin_type(builtin, Builtin_Type::UInt);
  exprt e = gen_zero(builtin);

  BOOST_CHECK_NO_THROW(gen_typecast_to_union(e, t));
  BOOST_TEST(to_union_expr(e).get_component_name() == "var_2");
  BOOST_TEST(to_union_expr(e).op0().type() == component2.type());
}

BOOST_AUTO_TEST_CASE(basic_union_5)
{
  auto component = gen_component("var_1", Builtin_Type::Double);
  auto component2 = gen_component("var_2", Builtin_Type::UInt);
  auto component3 = gen_component("var_3", Builtin_Type::UInt);

  union_typet t;
  t.components().push_back(component);
  t.components().push_back(component2);
  t.components().push_back(component3);

  typet builtin;
  gen_builtin_type(builtin, Builtin_Type::Double);
  exprt e = gen_zero(builtin);

  BOOST_CHECK_NO_THROW(gen_typecast_to_union(e, t));
  BOOST_TEST(to_union_expr(e).get_component_name() == "var_1");
  BOOST_TEST(to_union_expr(e).op0().type() == component.type());
}

BOOST_AUTO_TEST_SUITE_END()