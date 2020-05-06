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

#include <cache/containers/lru_cache.h>
#include <boost/test/included/unit_test.hpp>
namespace utf = boost::unit_test;

// ******************** TESTS ********************

// ** Constructors
// Check whether the object is initialized correctly

BOOST_AUTO_TEST_SUITE(constructors)
BOOST_AUTO_TEST_CASE(max_size_ok)
{
  lru_cache<int> obj(10);
  size_t expected = 10;
  size_t actual = obj.max_size();
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(max_size_fail)
{
  lru_cache<int> obj(10);
  size_t expected = 11;
  size_t actual = obj.max_size();
  BOOST_TEST(expected != actual);
}

BOOST_AUTO_TEST_CASE(initial_size_ok)
{
  lru_cache<int> obj(10);
  size_t expected = 0;
  size_t actual = obj.size();
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(element_doest_not_exist_ok)
{
  lru_cache<int> obj(10);
  bool expected = false;
  bool actual = obj.exists(0);
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_SUITE_END()

// ** Base methods
// Check whether the methods are behaving ok

BOOST_AUTO_TEST_SUITE(methods)

BOOST_AUTO_TEST_CASE(insertion_should_increase_size_ok_1)
{
  lru_cache<int> obj(10);
  obj.insert(0);
  int expected = 1;
  int actual = obj.size();
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(insertion_should_increase_size_ok_2)
{
  lru_cache<int> obj(10);
  obj.insert(0);
  obj.insert(1);
  int expected = 2;
  int actual = obj.size();
  BOOST_TEST(expected == actual);
}
BOOST_AUTO_TEST_CASE(insertion_should_respect_max_size_ok)
{
  lru_cache<int> obj(2);
  obj.insert(0);
  obj.insert(1);
  obj.insert(2);
  int expected = 2;
  int actual = obj.size();
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(insert_element_same_ok)
{
  lru_cache<int> obj(2);
  obj.insert(0);
  obj.insert(0);
  obj.insert(0);
  int expected = 1;
  int actual = obj.size();
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(element_in_list_ok_1)
{
  lru_cache<int> obj(2);
  obj.insert(0);
  bool expected = false;
  bool actual = obj.exists(2);
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(element_in_list_ok_2)
{
  lru_cache<int> obj(2);
  obj.insert(0);
  bool expected = true;
  bool actual = obj.exists(0);
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(element_in_list_ok_3)
{
  lru_cache<int> obj(2);
  obj.insert(0);
  obj.insert(1);
  obj.insert(2);
  bool expected = false;
  bool actual = obj.exists(0);
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_CASE(element_in_list_ok_4)
{
  lru_cache<int> obj(2);
  obj.insert(0);
  obj.insert(1);
  obj.insert(2);
  bool expected = true;
  bool actual = obj.exists(2);
  BOOST_TEST(expected == actual);
}
BOOST_AUTO_TEST_SUITE_END()

// ** Fuzzer cases
// Test cases that were found by libfuzzer

BOOST_AUTO_TEST_SUITE(libfuzzer)
BOOST_AUTO_TEST_CASE(crash_1)
{
  // This input was causing a memory leak because std::list is not
  // smart enough to know when it can release it's entries.
  const int input[] = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
  const size_t cache_size = 200;
  lru_cache<int> obj(cache_size);
  for(const auto &i : input)
  {
    obj.insert(i);
    BOOST_TEST(obj.exists(i));
  }
}

BOOST_AUTO_TEST_SUITE_END()