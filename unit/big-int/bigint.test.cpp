/*******************************************************************
 Module: BigInt unit test

 Author: Rafael Sá Menezes

 Date: December 2019

 Test Plan:
   - Basic usage scenarios
   - Template based tests
 \*******************************************************************/

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <big-int/bigint.hh>

const char *as_string(BigInt const &obj, std::vector<char> &vec)
{
  return obj.as_string(vec.data(), vec.size());
}

// ** Basic scenarios
// Check whether the object is initialized correctly

SCENARIO("bigint basic construction sets the correct values", "[bigint]")
{
  GIVEN("A bigint without a value")
  {
    BigInt obj;
    REQUIRE(obj.to_int64() == 0);
  }
  GIVEN("A bigint with some positive 64-bit value")
  {
    BigInt obj(42);
    REQUIRE(obj.to_int64() == 42);
  }
  GIVEN("A bigint with some negative 64-bit value")
  {
    BigInt obj(-42);
    REQUIRE(obj.to_int64() == -42);
  }
  GIVEN("A bigint with some string")
  {
    BigInt obj("-42", 10);
    REQUIRE(obj.to_int64() == -42);
  }
}

SCENARIO("bigint basic usage", "[bigint]")
{
  GIVEN("A bigint with a int64 value")
  {
    BigInt obj(42);
    WHEN("A new value is moved")
    {
      obj = -15;
      REQUIRE_FALSE(obj.to_int64() == 42);
      REQUIRE(obj.to_int64() == -15);
    }
    WHEN("Add is run")
    {
      obj += 10;
      REQUIRE(obj.to_int64() == 52);
    }
    WHEN("Sub is run")
    {
      obj -= 10;
      REQUIRE(obj.to_int64() == 32);
    }
    WHEN("Mul is run")
    {
      obj *= 10;
      REQUIRE(obj.to_int64() == 420);
    }
    WHEN("Div is run")
    {
      obj /= 10;
      REQUIRE(obj.to_int64() == 4);
    }
    WHEN("Neg is run")
    {
      obj = -obj;
      REQUIRE(obj.to_int64() == -42);
    }
  }
}

SCENARIO("bigint comparations", "[bigint]")
{
  GIVEN("Two bigints with int64 values")
  {
    BigInt A;
    BigInt B;

    WHEN("A is less than B")
    {
      A = 42;
      B = 100;

      REQUIRE(A != B);
      REQUIRE_FALSE(A == B);
      REQUIRE(A < B);
      REQUIRE_FALSE(B < A);
      REQUIRE(A <= B);
      REQUIRE_FALSE(B <= A);
      REQUIRE(B > A);
      REQUIRE_FALSE(A > B);
      REQUIRE(B >= A);
      REQUIRE_FALSE(A >= B);
    }

    WHEN("A is equal to B")
    {
      A = 42;
      B = 42;

      REQUIRE_FALSE(A != B);
      REQUIRE(A == B);
      REQUIRE_FALSE(A < B);
      REQUIRE_FALSE(B < A);
      REQUIRE(A <= B);
      REQUIRE(B <= A);
      REQUIRE_FALSE(B > A);
      REQUIRE_FALSE(A > B);
      REQUIRE(B >= A);
      REQUIRE(A >= B);
    }

    WHEN("A is greater than B")
    {
      A = 100;
      B = 42;

      REQUIRE(A != B);
      REQUIRE_FALSE(A == B);
      REQUIRE_FALSE(A < B);
      REQUIRE(B < A);
      REQUIRE_FALSE(A <= B);
      REQUIRE(B <= A);
      REQUIRE_FALSE(B > A);
      REQUIRE(A > B);
      REQUIRE_FALSE(B >= A);
      REQUIRE(A >= B);
    }
  }
}

/**
 * Next tests comes from CBMC, the only difference is that I
 * renamed some of the tags, I've removed tests that were dependent
 * on bigint support for doubles
 *
 * TODO: Add double tests
 *
 * Author: Daniel Kroening
 */

// =====================================================================
// Printing and reading bignums.
// =====================================================================

static std::string to_string(BigInt const &x, unsigned base = 10)
{
  const std::size_t len = x.digits(base) + 2;
  std::vector<char> dest(len, 0);
  const char *s = x.as_string(dest.data(), len, base);
  return std::string(s);
}

static bool read(const std::string &input, BigInt &x, unsigned base = 10)
{
  return x.scan(input.c_str(), base) == input.c_str() + input.size();
}

TEST_CASE("arbitrary precision integers", "[core][big-int][bigint]")
{
  // =====================================================================
  // Simple tests.
  // =====================================================================
  // Good when something basic is broken an must be debugged.
  SECTION("simple tests")
  {
    REQUIRE(to_string(BigInt(0xFFFFFFFFu)) == "4294967295");
    REQUIRE(
      to_string(BigInt(0xFFFFFFFFu), 2) == "11111111111111111111111111111111");
    REQUIRE(
      to_string(BigInt("123456789012345678901234567890")) ==
      "123456789012345678901234567890");

    REQUIRE(
      to_string(
        BigInt("99999999999999999999999999999999", 10) /
        BigInt("999999999999999999999999", 10)) == "100000000");
    REQUIRE(
      to_string(
        BigInt("99999999999999999999999999999999", 10) %
        BigInt("999999999999999999999999", 10)) == "99999999");

    BigInt t(100);
    t -= 300;
    REQUIRE(to_string(t) == "-200");

    BigInt r = BigInt(-124) + 124;
    REQUIRE(to_string(r) == "0");
    REQUIRE(BigInt(0) <= r);

    BigInt i(1);
    for (int j = 0; j < 1000; j++)
      i += 100000000;
    REQUIRE(to_string(i) == "100000000001");

    for (int j = 0; j < 2000; j++)
      i -= 100000000;
    REQUIRE(to_string(i) == "-99999999999");

    for (int j = 0; j < 1000; j++)
      i += 100000000;
    REQUIRE(to_string(i) == "1");
  }

  // =====================================================================
  // Test cases from the clisp test suite in number.tst.
  // =====================================================================

  // I took those test cases in number.tst from file
  //
  //  clisp-1998-09-09/tests/number.tst
  //
  // in clispsrc.tar.gz. From the README file in that directory:
  /*

  This directory contains a test suite for testing Common Lisp (CLtL1)
  implementations.

  In its original version it was built by

      Horst Friedrich, ISST of FhG         <horst.friedrich@isst.fhg.de>
      Ingo Mohr, ISST of FhG               <ingo.mohr@isst.fhg.de>
      Ulrich Kriegel, ISST of FhG          <ulrich.kriegel@isst.fhg.de>
      Windfried Heicking, ISST of FhG      <winfried.heicking@isst.fhg.de>
      Rainer Rosenmueller, ISST of FhG     <rainer.rosenmueller@isst.fhg.de>

  at

      Institut für Software- und Systemtechnik der Fraunhofer-Gesellschaft
      (Fraunhofer Institute for Software Engineering and Systems Engineering)
      Kurstraße 33
    D-10117 Berlin
      Germany

  for their Common Lisp implementation named XCL.

  What you see here is a version adapted to CLISP and AKCL by

      Bruno Haible              <haible@ma2s2.mathematik.uni-karlsruhe.de>
  */

  // Actually I have no idea what principles directed the choice of test
  // cases and what they are worth. Nevertheless it makes me feel better
  // when BigInt comes to the same results as a Common Lisp should. Note
  // that Lisp uses a floored divide operator which means that the
  // quotient is rounded towards negative infinity. The remainder has to
  // be adjusted accordingly.

  // Each test is operator op1 op2 result [result2]. Everything is white
  // space delimited with line breaks meaning nothing special. Read
  // operator and operands, compute, compare with expected result and
  // complain if not.
  SECTION("clisp tests")
  {
    const std::vector<std::string> number_tst = {
#include "number.tst"
    };

    for (std::size_t i = 0; i < number_tst.size(); i += 4)
    {
      const std::string op = number_tst[i];
      REQUIRE(!op.empty());

      BigInt a, b, r, er;
      REQUIRE(read(number_tst[i + 1], a));
      REQUIRE(read(number_tst[i + 2], b));
      REQUIRE(read(number_tst[i + 3], er));

      switch (op[0])
      {
      case '+':
        r = a + b;
        REQUIRE(r == er);
        break;
      case '-':
        r = a - b;
        REQUIRE(r == er);
        break;
      case '*':
        r = a * b;
        REQUIRE(r == er);
        break;
      case '/':
      {
        // These lines also have a remainder.
        REQUIRE(i + 4 < number_tst.size());
        BigInt em;
        REQUIRE(read(number_tst[i + 4], em));
        ++i;

        r = a / b;
        BigInt m = a % b;
        // The test-data from the Lisp testsuite are assuming
        // floored divide. Fix the results accordingly.
        if (!m.is_zero() && a.is_positive() != b.is_positive())
        {
          r -= 1;
          m += b;
        }
        REQUIRE(r == er);
        REQUIRE(m == em);

        // Also try the method returning both.
        BigInt::div(a, b, r, m);
        // Again, transform to floored divide.
        if (!m.is_zero() && a.is_positive() != b.is_positive())
        {
          r -= 1;
          m += b;
        }
        REQUIRE(r == er);
        REQUIRE(m == em);
      }
      }
    }
  }
}
