/*******************************************************************
 Module: LRU Cache unit test

 Author: Rafael Sá Menezes

 Date: January 2019

 Test Plan:
   - Constructors
   - Base Methods
   - Integrity/Atomicity
 \*******************************************************************/

#define BOOST_TEST_MODULE "LRU Cache"

#include <util/lru_cache.h>
#include <boost/test/included/unit_test.hpp>
namespace utf = boost::unit_test;

// ******************** TESTS ********************

// ** Constructors
// Check whether the object is initialized correctly

BOOST_AUTO_TEST_SUITE(constructors)
BOOST_AUTO_TEST_CASE(max_size_ok)
{
  lru_cache<int, int> obj(10);
  size_t expected = 10;
  size_t actual = obj.max_size();
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(max_size_fail)
{
  lru_cache<int, int> obj(10);
  size_t expected = 11;
  size_t actual = obj.max_size();
  BOOST_TEST(expected != actual);
}

BOOST_AUTO_TEST_CASE(initial_size_ok)
{
  lru_cache<int, int> obj(10);
  size_t expected = 0;
  size_t actual = obj.size();
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(element_doest_not_exist_ok)
{
  lru_cache<int, int> obj(10);
  bool expected = false;
  bool actual = obj.exists(0);
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(get_should_throw_exception)
{
  lru_cache<int, int> obj(10);
  BOOST_CHECK_THROW(obj.get(0), std::range_error);
}

BOOST_AUTO_TEST_SUITE_END()

// ** Base methods
// Check whether the methods are behaving ok

BOOST_AUTO_TEST_SUITE(methods)

BOOST_AUTO_TEST_CASE(insertion_should_increase_size_ok_1)
{
  lru_cache<int, int> obj(10);
  obj.insert(0, 1);
  int expected = 1;
  int actual = obj.size();
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(insertion_should_increase_size_ok_2)
{
  lru_cache<int, int> obj(10);
  obj.insert(0, 1);
  obj.insert(1, 1);
  int expected = 2;
  int actual = obj.size();
  BOOST_TEST(expected == actual);
}
BOOST_AUTO_TEST_CASE(insertion_should_respect_max_size_ok)
{
  lru_cache<int, int> obj(2);
  obj.insert(0, 1);
  obj.insert(1, 1);
  obj.insert(2, 2);
  int expected = 2;
  int actual = obj.size();
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(insert_element_same_ok)
{
  lru_cache<int, int> obj(2);
  obj.insert(0, 1);
  obj.insert(0, 2);
  obj.insert(0, 3);
  int expected = 1;
  int actual = obj.size();
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(element_in_list_ok_1)
{
  lru_cache<int, int> obj(2);
  obj.insert(0, 1);
  bool expected = false;
  bool actual = obj.exists(2);
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(element_in_list_ok_2)
{
  lru_cache<int, int> obj(2);
  obj.insert(0, 1);
  bool expected = true;
  bool actual = obj.exists(0);
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(element_in_list_ok_3)
{
  lru_cache<int, int> obj(2);
  obj.insert(0, 1);
  obj.insert(1, 2);
  obj.insert(2, 3);
  bool expected = false;
  bool actual = obj.exists(0);
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(element_in_list_ok_4)
{
  lru_cache<int, int> obj(2);
  obj.insert(0, 1);
  obj.insert(1, 2);
  obj.insert(2, 3);
  bool expected = true;
  bool actual = obj.exists(2);
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(get_element_ok_1)
{
  lru_cache<int, int> obj(2);
  obj.insert(0, 1);
  obj.insert(1, 2);

  int expected = 2;
  int actual = obj.get(1);
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(get_element_fail_1)
{
  lru_cache<int, int> obj(2);
  obj.insert(0, 1);
  obj.insert(1, 2);
  obj.insert(2, 3);
  BOOST_CHECK_THROW(obj.get(0), std::range_error);
}
BOOST_AUTO_TEST_SUITE_END()