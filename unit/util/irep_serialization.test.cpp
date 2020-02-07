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
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>


template <class T>
void write_load_serialization(T &original, T &copy) {
  std::ofstream ofs("filename");

  // save data to archive
  {
    boost::archive::text_oarchive oa(ofs);
    // write class instance to archive
    oa << original;
    // // archive and stream closed when destructors are called
  }

  {
    std::ifstream ifs("filename");
    boost::archive::text_iarchive ia(ifs);
    // read class state from archive
    ia >> copy;
  }
}

// ******************** TESTS ********************

// ** Constructors
// Check whether the object is initialized correctly

BOOST_AUTO_TEST_SUITE(serializaton)
BOOST_AUTO_TEST_CASE(write_read_long)
{
  unsigned expected = 1234;
  unsigned actual;
  std::ostringstream stream;
  irep_serializationt::write_long(stream, 1234);
  std::istringstream instream;
  instream.str(stream.str());
  actual = irep_serializationt::read_long(instream);

  BOOST_CHECK_EQUAL(expected, actual);
}

BOOST_AUTO_TEST_CASE(write_read_string)
{
  std::string expected = "1234";
  std::string actual;
  std::ostringstream stream;
  irep_serializationt::write_string(stream, "1234");
  std::istringstream instream;
  instream.str(stream.str());
  actual = irep_serializationt::read_string(instream).as_string();
  BOOST_CHECK_EQUAL(expected, actual);
}

BOOST_AUTO_TEST_CASE(write_read_dstring)
{
  dstring expected("asd"), actual("qwe");
  write_load_serialization(expected, actual);
  BOOST_CHECK_EQUAL(expected, actual);
}

BOOST_AUTO_TEST_CASE(write_read_bool_type2t)
{
  bool_type2t expected, actual;
  expected.type_id = type2t::type_ids::floatbv_id;
  write_load_serialization(expected, actual);
  BOOST_CHECK_EQUAL(expected.type_id, actual.type_id);
}

BOOST_AUTO_TEST_CASE(write_read_empty_type2t)
{
  empty_type2t expected, actual;
  expected.type_id = type2t::type_ids::floatbv_id;
  write_load_serialization(expected, actual);
  BOOST_CHECK_EQUAL(expected.type_id, actual.type_id);
}

BOOST_AUTO_TEST_CASE(write_read_symbol_type2t)
{
  symbol_type2t expected("asd"), actual("qwe");
  write_load_serialization(expected, actual);
  BOOST_CHECK_EQUAL(expected.symbol_name, actual.symbol_name);
}

BOOST_AUTO_TEST_CASE(write_read_unsignedbv_type2t)
{
  unsignedbv_type2t expected(7), actual(9);
  write_load_serialization(expected, actual);
  BOOST_CHECK_EQUAL(expected.width, actual.width);
}

BOOST_AUTO_TEST_CASE(write_read_signedbv_type2t)
{
  signedbv_type2t expected(7), actual(9);
  write_load_serialization(expected, actual);
  BOOST_CHECK_EQUAL(expected.width, actual.width);
}

BOOST_AUTO_TEST_SUITE_END()