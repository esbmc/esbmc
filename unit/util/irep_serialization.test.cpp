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
#include <boost/test/included/unit_test.hpp>

// ******************** TESTS ********************

// ** Constructors
// Check whether the object is initialized correctly

BOOST_AUTO_TEST_SUITE(serializaton)
BOOST_AUTO_TEST_CASE(write_long_buffer)
{
  unsigned expected = 1234;
  unsigned actual;
  std::ostringstream stream;
  write_long(stream, 1234);
  std::istringstream instream;
  instream.str(stream.str());
  actual = irep_serializationt::read_long(instream);

  BOOST_CHECK_EQUAL(expected, actual);
}

BOOST_AUTO_TEST_CASE(write_string_buffer)
{
  std::string expected = "1234";
  std::string actual;
  std::ostringstream stream;
  write_string(stream, "1234");
  std::istringstream instream;
  instream.str(stream.str());
  actual = irep_serializationt::read_string(instream).as_string();
  BOOST_CHECK_EQUAL(expected, actual);
}

BOOST_AUTO_TEST_SUITE_END()