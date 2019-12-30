/*******************************************************************
 Module: BigInt unit test

 Author: Rafael Sá Menezes

 Date: December 2019

 Test Plan:
   - Constructors
   - Assignments
   - Comparations
   - Math Operations
   - Class Helpers
 \*******************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "Big Int"

#include <big-int/bigint.hh>
#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;

namespace {
const char *as_string(BigInt const &obj, std::vector<char> &vec) {
  return obj.as_string(vec.data(), vec.size());
}

void check_bigint_str(BigInt const &obj, const char *expected,
                      bool is_correct) {
  std::vector<char> bigint_str(obj.digits());
  const char *actual = as_string(obj, bigint_str);
  if (is_correct)
    BOOST_TEST(expected == actual);
  else
    BOOST_TEST(expected != actual);
}

template <class T> struct BigIntHelper {
  BigInt obj;

  BigIntHelper() {}
  explicit BigIntHelper(T val) : obj(val) {}
  BigIntHelper(char const *val, BigInt::onedig_t base) : obj(val, base) {}
  void check_value(const char *expected, bool is_correct = true) {
    check_bigint_str(obj, expected, is_correct);
  }
};
} // namespace

// ******************** TESTS ********************

// ** Constructors
// Check whether the object is initialized correctly
BOOST_AUTO_TEST_CASE(null_constructor_ok) {
  BigIntHelper<int> obj;
  obj.check_value(NULL);
}
BOOST_AUTO_TEST_CASE(null_constructor_fail) {
  BigIntHelper<int> obj;
  obj.check_value("32", false);
}

BOOST_AUTO_TEST_CASE(signed_constructor_ok) {
  const int input = -42;
  BigIntHelper<int> obj(input);
  obj.check_value("-42");
}
BOOST_AUTO_TEST_CASE(signed_constructor_fail) {
  const int input = -42;
  BigIntHelper<int> obj(input);
  obj.check_value("32", false);
}

BOOST_AUTO_TEST_CASE(unsigned_constructor_ok) {
  const unsigned input = 398;
  BigIntHelper<int> obj(input);
  obj.check_value("398");
}
BOOST_AUTO_TEST_CASE(unsigned_constructor_fail) {
  const unsigned input = 0;
  BigIntHelper<int> obj(input);
  obj.check_value("398", false);
}

BOOST_AUTO_TEST_CASE(string_constructor_ok_1) {
  const BigInt::onedig_t base = 10;
  const char *input = "42";
  BigIntHelper<int> obj(input, base);
  obj.check_value("42");
}
BOOST_AUTO_TEST_CASE(string_constructor_fail_1) {
  const BigInt::onedig_t base = 10;
  const char *input = "42";
  BigIntHelper<int> obj(input, base);
  obj.check_value("79", false);
}

BOOST_AUTO_TEST_CASE(string_constructor_ok_2) {
  const BigInt::onedig_t base = 16;
  const char *input = "FF";
  BigIntHelper<int> obj(input, base);
  obj.check_value("255");
}
BOOST_AUTO_TEST_CASE(string_constructor_fail_2) {
  const BigInt::onedig_t base = 16;
  const char *input = "A";
  BigIntHelper<int> obj(input, base);
  obj.check_value("100", false);
}

BOOST_AUTO_TEST_CASE(string_constructor_ok_3) {
  const BigInt::onedig_t base = 10;
  const char *input = "12345678987654321234567890";
  BigIntHelper<int> obj(input, base);
  obj.check_value("12345678987654321234567890");
}
BOOST_AUTO_TEST_CASE(string_constructor_fail_3) {
  const BigInt::onedig_t base = 10;
  const char *input = "12345678987654321234567890";
  BigIntHelper<int> obj(input, base);
  obj.check_value("1234567898765432123456789", false);
}

BOOST_AUTO_TEST_CASE(bigint_constructor_ok) {
  BigInt input(42);
  BigIntHelper<BigInt> obj(input);
  obj.check_value("42");
}
BOOST_AUTO_TEST_CASE(bigint_constructor_fail) {
  BigInt input(42);
  BigIntHelper<BigInt> obj(input);
  obj.check_value(NULL, false);
}

// ** Assignment
// Check whether the object is initialized correctly after an assignement

BOOST_AUTO_TEST_CASE(signed_assignment_ok) {
  BigIntHelper<unsigned> obj(42);
  obj.obj = -9090;
  obj.check_value("-9090");
}
BOOST_AUTO_TEST_CASE(signed_assignment_fail) {
  BigIntHelper<int> obj(-42);
  obj.obj = -9090;
  obj.check_value("-42", false);
}

BOOST_AUTO_TEST_CASE(unsigned_assignment_ok) {
  BigIntHelper<int> obj(-255);
  unsigned value = 400000;
  obj.obj = value;
  obj.check_value("400000");
}
BOOST_AUTO_TEST_CASE(unsigned_assignment_fail) {
  BigIntHelper<int> obj(40000);
  obj.obj = 9090;
  obj.check_value("40000", false);
}

BOOST_AUTO_TEST_CASE(bigint_assignment_ok) {
  BigIntHelper<int> obj(-255);
  BigInt value(42);
  obj.obj = value;
  obj.check_value("42");
}
BOOST_AUTO_TEST_CASE(bigint_assignment_fail) {
  BigInt value(42);
  BigIntHelper<BigInt> obj(value);
  obj.obj = 9090;
  obj.check_value("42", false);
}

BOOST_AUTO_TEST_CASE(string_assignment_ok_1) {
  const char *input = "12345678987654321234567890";
  BigIntHelper<int> obj(-255);
  obj.obj = input;
  obj.check_value(input);
}

BOOST_AUTO_TEST_CASE(string_assignment_ok_2) {
  BigIntHelper<int> obj(-255);
  obj.obj = "255";
  obj.check_value("255");
}

BOOST_AUTO_TEST_CASE(string_assignment_ok_3) {
  BigIntHelper<int> obj(255);
  obj.obj = "-255";
  obj.check_value("-255");
}

BOOST_AUTO_TEST_CASE(string_assignment_ok_4) {
  const BigInt::onedig_t base = 10;
  const char *input = "12345678987654321234567890";
  BigIntHelper<char> obj(input, base);
  obj.obj = input;
  obj.check_value(input);
}

/*
BOOST_AUTO_TEST_CASE(string_assignment_fail) {
  const BigInt::onedig_t base = 10;
  const char *input = "12345678987654321234567890";
  BigIntHelper<int> obj(input, base);
  obj.obj = -255;
  obj.check_value("-255", false);
}
*/

BOOST_AUTO_TEST_CASE(string_assignment_comparation_ok) {
  const char *input = "12345678987654321234567890";
  BigInt obj_1(-255);
  obj_1 = input;

  BigInt obj_2(input);
  BOOST_TEST(obj_1.compare(obj_2));
}