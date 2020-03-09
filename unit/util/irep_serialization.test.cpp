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
#include <util/migrate.h>

namespace btf = boost::unit_test::framework;
// THIS APPROACH IS NOT THREAD SAFE!!!!
void write_serialization(const type2tc &original, const std::string &file_name)
{
  std::ofstream ofs(file_name, std::ofstream::binary);
  irep_serializationt::write_irep2(ofs, original);
}

std::unique_ptr<std::istream> load_serialization(const std::string &file_name)
{
  auto ptr = std::make_unique<std::ifstream>(file_name, std::ifstream::binary);
  return std::move(ptr);
}

void init_type_pool()
{
  static bool initialized = false;
  if(!initialized)
  {
    type_poolt hack(true);
    type_pool = hack;
    initialized = true;
  }
}

void type2t_types_test(const type2tc &irep2, type2t::type_ids id)
{
  std::string file_name(btf::current_test_case().p_name);
  write_serialization(irep2, file_name);
  auto istream = load_serialization(file_name);
  typet irep;
  irep_serializationt::read_irep(*istream, irep);
  type2tc irep2_in;
  migrate_type(irep, irep2_in);
  BOOST_CHECK_EQUAL(irep2->type_id, id);
  BOOST_CHECK_EQUAL(irep2->type_id, irep2_in->type_id);
}

#define type2t_int_test(SIZE)                                                  \
  init_type_pool();                                                            \
  const type2tc &irep2 = type_pool.get_int##SIZE();                            \
  type2t_types_test(irep2, type2t::type_ids::signedbv_id);

#define type2t_uint_test(SIZE)                                                 \
  init_type_pool();                                                            \
  const type2tc &irep2 = type_pool.get_uint##SIZE();                           \
  type2t_types_test(irep2, type2t::type_ids::unsignedbv_id);

#define type2t_integer_generator(SIZE)                                         \
  BOOST_AUTO_TEST_CASE(type2t_uint##SIZE){                                     \
    type2t_uint_test(SIZE)} BOOST_AUTO_TEST_CASE(type2t_int##SIZE)             \
  {                                                                            \
    type2t_int_test(SIZE)                                                      \
  }

// ******************** TESTS ********************

// ** TypeID
// Check whether the object type is reconstructed correctly
BOOST_AUTO_TEST_SUITE(type_id_check)

BOOST_AUTO_TEST_CASE(type2t_bool_type)
{
  init_type_pool();
  const type2tc &irep2 = type_pool.get_bool();
  type2t_types_test(irep2, type2t::type_ids::bool_id);
}

BOOST_AUTO_TEST_CASE(type2t_empty_type)
{
  init_type_pool();
  const type2tc &irep2 = type_pool.get_empty();
  type2t_types_test(irep2, type2t::type_ids::empty_id);
}

BOOST_AUTO_TEST_CASE(type2t_struct_type)
{
  init_type_pool();
  const std::vector<type2tc> members;
  const std::vector<irep_idt> memb_names;
  const std::vector<irep_idt> memb_pretty_names;
  const irep_idt name;
  const type2tc &irep2 =
    struct_type2tc(members, memb_names, memb_pretty_names, name);
  type2t_types_test(irep2, type2t::type_ids::struct_id);
}

BOOST_AUTO_TEST_CASE(type2t_union_type)
{
  init_type_pool();
  const std::vector<type2tc> members;
  const std::vector<irep_idt> memb_names;
  const std::vector<irep_idt> memb_pretty_names;
  const irep_idt name("name"); // Initialization is Required
  const type2tc &irep2 =
    union_type2tc(members, memb_names, memb_pretty_names, name);
  type2t_types_test(irep2, type2t::type_ids::union_id);
}

BOOST_AUTO_TEST_CASE(type2t_array_type)
{
  init_type_pool();
  const type2tc &_subtype = type_pool.get_bool(); // Initialization is Required
  const expr2tc size;
  bool inf;
  const type2tc &irep2 = array_type2tc(_subtype, size, inf);
  type2t_types_test(irep2, type2t::type_ids::array_id);
}

BOOST_AUTO_TEST_CASE(type2t_pointer_type)
{
  init_type_pool();
  const type2tc &_subtype = type_pool.get_uint8(); // Initialization is Required
  const type2tc irep2(new pointer_type2t(_subtype));
  type2t_types_test(irep2, type2t::type_ids::pointer_id);
}

BOOST_AUTO_TEST_CASE(type2t_string_type)
{
  init_type_pool();
  const unsigned int elements = 16; // Any number greater than 0
  const type2tc &irep2 = string_type2tc(elements);
  type2t_types_test(irep2, type2t::type_ids::string_id);
}

BOOST_AUTO_TEST_CASE(type2t_floatbv_type)
{
  init_type_pool();
  const unsigned int fraction = 2; // Any number greater than 0
  const unsigned int exponent = 4; // Any number greater than 0
  const type2tc &irep2 = floatbv_type2tc(fraction, exponent);
  type2t_types_test(irep2, type2t::type_ids::floatbv_id);
}

BOOST_AUTO_TEST_CASE(type2t_fixedbv_type)
{
  init_type_pool();
  const unsigned int width = 2; // Any number greater than 0
  const unsigned int integer = 4; // Any number greater than 0
  const type2tc &irep2 = fixedbv_type2tc(width, integer);
  type2t_types_test(irep2, type2t::type_ids::fixedbv_id);
}

BOOST_AUTO_TEST_CASE(type2t_code_type)
{
  init_type_pool();
  const std::vector<type2tc> args;
  const type2tc &ret_type = type_pool.get_int8();
  const std::vector<irep_idt> names;
  const bool e = true;
  const type2tc &irep2 = code_type2tc(args, ret_type, names, e);
  type2t_types_test(irep2, type2t::type_ids::code_id);
}

BOOST_AUTO_TEST_CASE(type2t_symbol_type)
{
  init_type_pool();
  const dstring sym_name("name"); // Initialization is Required
  const type2tc &irep2 = symbol_type2tc(sym_name);
  type2t_types_test(irep2, type2t::type_ids::symbol_id);
}

type2t_integer_generator(8);
type2t_integer_generator(16);
type2t_integer_generator(32);
type2t_integer_generator(64);

BOOST_AUTO_TEST_SUITE_END();
