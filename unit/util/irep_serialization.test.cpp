/*******************************************************************
 Module: irep serialization unit test

 Author: Rafael Sá Menezes

 Date: February 2020

 Test Plan:
   - Constructors
   - Conversions
 \*******************************************************************/

#define BOOST_TEST_MODULE "irep Serialization"

#include <util/irep_serialization.h>
#include <util/irep2.h>
#include <util/irep2_type.h>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/framework.hpp>
#include <util/irep2_expr.h>

namespace btf = boost::unit_test::framework;
// THIS APPROACH IS NOT THREAD SAFE!!!!
void write_serialization(
  irep_serializable &original,
  const std::string &file_name)
{
  std::ofstream ofs(file_name, std::ofstream::binary);
  original.serialize(ofs);
}

std::unique_ptr<std::istream> load_serialization(const std::string &file_name)
{
  auto ptr = std::make_unique<std::ifstream>(file_name, std::ifstream::binary);
  return std::move(ptr);
}

// ******************** TESTS ********************

// ** type deduction
// Check whether the object is reconstructed with correct type

#define type_deduction_helper(TYPE)                                            \
  std::string file_name(btf::current_test_case().p_name);                      \
  TYPE obj;                                                                    \
  write_serialization(obj, file_name);                                         \
  auto istream = load_serialization(file_name);

#define type_deduction_ok(OK_TYPE)                                             \
  type_deduction_helper(OK_TYPE)                                               \
    std::static_pointer_cast<OK_TYPE>(OK_TYPE::unserialize(*istream));

#define type_deduction_false(OK_TYPE, WRONG_TYPE)                              \
  type_deduction_helper(WRONG_TYPE)                                            \
    BOOST_CHECK_THROW(OK_TYPE::unserialize(*istream), std::bad_cast);

#define type_deduction_generator(TYPE, WRONG_TYPE)                             \
  BOOST_AUTO_TEST_CASE(ok_##TYPE){type_deduction_ok(TYPE)};                    \
  BOOST_AUTO_TEST_CASE(bad_##TYPE){type_deduction_false(TYPE, WRONG_TYPE)};

#define type_deduction_type2t_ok(OK_TYPE, ID)                                  \
  type_deduction_helper(OK_TYPE);                                              \
  auto check =                                                                 \
    std::dynamic_pointer_cast<OK_TYPE>(type2t::unserialize(*istream));         \
  BOOST_CHECK_EQUAL(check->type_id, ID);

#define type_deduction_type2t(OK_TYPE, WRONG_TYPE, ID)                         \
  BOOST_AUTO_TEST_CASE(ok_##OK_TYPE){type_deduction_type2t_ok(OK_TYPE, ID)};

BOOST_AUTO_TEST_SUITE(type_deduction)

// Containers
type_deduction_generator(type2tc, expr2tc);
type_deduction_generator(expr2tc, type2tc);

// type2t
type_deduction_type2t(bool_type2t, empty_type2t, type2t::type_ids::bool_id);
type_deduction_type2t(empty_type2t, bool_type2t, type2t::type_ids::empty_id);

BOOST_AUTO_TEST_CASE(ok_cpp_name_type2t)
{
  std::string file_name(btf::current_test_case().p_name);
  irep_idt irep;
  std::vector<type2tc> vec;
  cpp_name_type2t obj(irep, vec);
  write_serialization(obj, file_name);
  auto istream = load_serialization(file_name);
  auto check =
    std::dynamic_pointer_cast<cpp_name_type2t>(type2t::unserialize(*istream));
  BOOST_CHECK_EQUAL(check->type_id, type2t::type_ids::cpp_name_id);
}

BOOST_AUTO_TEST_CASE(ok_string_type2t)
{
  std::string file_name(btf::current_test_case().p_name);
  string_type2t obj(0);
  write_serialization(obj, file_name);
  auto istream = load_serialization(file_name);
  auto check =
    std::dynamic_pointer_cast<string_type2t>(type2t::unserialize(*istream));
  BOOST_CHECK_EQUAL(check->type_id, type2t::type_ids::string_id);
}

BOOST_AUTO_TEST_CASE(ok_floatbv_type2t)
{
  std::string file_name(btf::current_test_case().p_name);
  floatbv_type2t obj(0, 1);
  write_serialization(obj, file_name);
  auto istream = load_serialization(file_name);
  auto check =
    std::dynamic_pointer_cast<floatbv_type2t>(type2t::unserialize(*istream));
  BOOST_CHECK_EQUAL(check->type_id, type2t::type_ids::floatbv_id);
}

BOOST_AUTO_TEST_CASE(ok_fixedbv_type2t)
{
  std::string file_name(btf::current_test_case().p_name);
  fixedbv_type2t obj(0, 1);
  write_serialization(obj, file_name);
  auto istream = load_serialization(file_name);
  auto check =
    std::dynamic_pointer_cast<fixedbv_type2t>(type2t::unserialize(*istream));
  BOOST_CHECK_EQUAL(check->type_id, type2t::type_ids::fixedbv_id);
}

BOOST_AUTO_TEST_CASE(ok_pointer_type2t)
{
  std::string file_name(btf::current_test_case().p_name);
  type2tc tc;
  pointer_type2t obj(tc);
  write_serialization(obj, file_name);
  auto istream = load_serialization(file_name);
  auto check =
    std::dynamic_pointer_cast<pointer_type2t>(type2t::unserialize(*istream));
  BOOST_CHECK_EQUAL(check->type_id, type2t::type_ids::pointer_id);
}

BOOST_AUTO_TEST_CASE(ok_array_type2t)
{
  std::string file_name(btf::current_test_case().p_name);
  type2tc tc;
  expr2tc ec;
  array_type2t obj(tc, ec, true);
  write_serialization(obj, file_name);
  auto istream = load_serialization(file_name);
  auto check =
    std::dynamic_pointer_cast<array_type2t>(type2t::unserialize(*istream));
  BOOST_CHECK_EQUAL(check->type_id, type2t::type_ids::array_id);
}

BOOST_AUTO_TEST_CASE(ok_code_type2t)
{
  std::string file_name(btf::current_test_case().p_name);
  std::vector<type2tc> vec_code_type2t;
  type2tc tc_code_type2t;
  std::vector<irep_idt> names_code_type2t;
  code_type2t obj(vec_code_type2t, tc_code_type2t, names_code_type2t, true);
  write_serialization(obj, file_name);
  auto istream = load_serialization(file_name);
  auto check =
    std::dynamic_pointer_cast<code_type2t>(type2t::unserialize(*istream));
  BOOST_CHECK_EQUAL(check->type_id, type2t::type_ids::code_id);
}

BOOST_AUTO_TEST_CASE(ok_signedbv_type2t)
{
  std::string file_name(btf::current_test_case().p_name);
  signedbv_type2t obj(0);
  write_serialization(obj, file_name);
  auto istream = load_serialization(file_name);
  auto check =
    std::dynamic_pointer_cast<signedbv_type2t>(type2t::unserialize(*istream));
  BOOST_CHECK_EQUAL(check->type_id, type2t::type_ids::signedbv_id);
}

BOOST_AUTO_TEST_CASE(ok_unsignedbv_type2t)
{
  std::string file_name(btf::current_test_case().p_name);
  unsignedbv_type2t obj(0);
  write_serialization(obj, file_name);
  auto istream = load_serialization(file_name);
  auto check =
    std::dynamic_pointer_cast<unsignedbv_type2t>(type2t::unserialize(*istream));
  BOOST_CHECK_EQUAL(check->type_id, type2t::type_ids::unsignedbv_id);
}

BOOST_AUTO_TEST_CASE(ok_union_type2t)
{
  std::string file_name(btf::current_test_case().p_name);
  const std::vector<type2tc> tc_union_type2t;
  const std::vector<irep_idt> irep_union_type2t;
  const std::vector<irep_idt> irep2_union_type2t;
  const irep_idt name_union_type2t;
  union_type2t obj(
    tc_union_type2t, irep_union_type2t, irep2_union_type2t, name_union_type2t);
  write_serialization(obj, file_name);
  auto istream = load_serialization(file_name);
  auto check =
    std::dynamic_pointer_cast<union_type2t>(type2t::unserialize(*istream));
  BOOST_CHECK_EQUAL(check->type_id, type2t::type_ids::union_id);
}

BOOST_AUTO_TEST_CASE(ok_struct_type2t)
{
  std::string file_name(btf::current_test_case().p_name);
  const std::vector<type2tc> tc_union_type2t;
  const std::vector<irep_idt> irep_union_type2t;
  const std::vector<irep_idt> irep2_union_type2t;
  const irep_idt name_union_type2t;
  struct_type2t obj(
    tc_union_type2t, irep_union_type2t, irep2_union_type2t, name_union_type2t);
  write_serialization(obj, file_name);
  auto istream = load_serialization(file_name);
  auto check =
    std::dynamic_pointer_cast<struct_type2t>(type2t::unserialize(*istream));
  BOOST_CHECK_EQUAL(check->type_id, type2t::type_ids::struct_id);
}

BOOST_AUTO_TEST_CASE(ok_symbol_type2t)
{
  std::string file_name(btf::current_test_case().p_name);
  symbol_type2t obj("asd");
  write_serialization(obj, file_name);
  auto istream = load_serialization(file_name);
  auto check =
    std::dynamic_pointer_cast<symbol_type2t>(type2t::unserialize(*istream));
  BOOST_CHECK_EQUAL(check->type_id, type2t::type_ids::symbol_id);
}

// expr2t

BOOST_AUTO_TEST_CASE(ok_unknown2t)
{
  std::string file_name(btf::current_test_case().p_name);
  type2tc tc;
  unknown2t obj(tc);
  write_serialization(obj, file_name);
  auto istream = load_serialization(file_name);
  auto check =
    std::dynamic_pointer_cast<unknown2t>(expr2t::unserialize(*istream));
  BOOST_CHECK_EQUAL(check->expr_id, expr2t::expr_ids::unknown_id);
}

BOOST_AUTO_TEST_SUITE_END();