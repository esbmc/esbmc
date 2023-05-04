#include <goto-programs/abstract-interpretation/interval_template.h>
#include <goto-programs/abstract-interpretation/wrapped_interval.h>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "c_types.h"

TEST_CASE(
  "Interval templates base functions are working for int",
  "[ai][interval-analysis]")
{
  interval_templatet<int> A;
  SECTION("Initialization tests")
  {
    REQUIRE(A.is_top());
    REQUIRE_FALSE(A.is_bottom());
    REQUIRE_FALSE(A.singleton());
  }
  SECTION("Lower bound is set")
  {
    A.make_ge_than(-3);
    REQUIRE_FALSE(A.is_top());
    REQUIRE_FALSE(A.is_bottom());
    REQUIRE_FALSE(A.singleton());
  }
  SECTION("Upper bound is set (singleton)")
  {
    A.make_ge_than(-3);
    A.make_le_than(0);
    REQUIRE_FALSE(A.is_top());
    REQUIRE_FALSE(A.is_bottom());
    REQUIRE_FALSE(A.singleton());
  }
  SECTION("Upper bound is set (singleton)")
  {
    A.make_ge_than(-3);
    A.make_le_than(-3);
    REQUIRE_FALSE(A.is_top());
    REQUIRE_FALSE(A.is_bottom());
    REQUIRE(A.singleton());
  }
  SECTION("Contradiction in set")
  {
    A.make_ge_than(0);
    A.make_le_than(-1);
    REQUIRE_FALSE(A.is_top());
    REQUIRE(A.is_bottom());
    REQUIRE_FALSE(A.singleton());
  }
}

TEST_CASE(
  "Interval templates relation functions are working for int",
  "[ai][interval-analysis]")
{
  interval_templatet<int> A;
  interval_templatet<int> B;

  SECTION("Union between two tops is a top")
  {
    REQUIRE(A.is_top());
    REQUIRE(B.is_top());
    A.join(B);
    REQUIRE(A.is_top());
  }

  SECTION("Union with a top results in a top")
  {
    A.make_ge_than(0);
    REQUIRE_FALSE(A.is_top());
    REQUIRE(B.is_top());
    A.join(B);
    REQUIRE(A.is_top());
  }

  SECTION("Intersection with a top results in itself")
  {
    A.make_ge_than(0);
    REQUIRE_FALSE(A.is_top());
    interval_templatet<int> previous = A;
    REQUIRE(B.is_top());
    A.meet(B);
    REQUIRE(A == previous);
  }

  SECTION("Union with a bottom results in itself")
  {
    A.make_ge_than(0);
    REQUIRE_FALSE(A.is_top());
    interval_templatet<int> previous = A;

    B.make_ge_than(1);
    B.make_le_than(0);
    REQUIRE(B.is_bottom());
    A.join(B);
    REQUIRE(A == previous);
  }

  SECTION("Intersect with a bottom results in a bottom")
  {
    REQUIRE(A.is_top());

    B.make_ge_than(1);
    B.make_le_than(0);
    REQUIRE(B.is_bottom());
    A.meet(B);
    REQUIRE(A.is_bottom());
  }

  SECTION("Union of two sets should over-aproximate")
  {
    // [0,4] U [10,20] = [0,20]
    A.make_ge_than(0);
    A.make_le_than(4);

    B.make_ge_than(10);
    B.make_le_than(20);

    interval_templatet<int> expected;
    expected.make_ge_than(0);
    expected.make_le_than(20);

    A.approx_union_with(B);
    REQUIRE(A == expected);
  }

  SECTION("Intersect should... intersect")
  {
    // [0,15] intersec [10,20] = [10,15]
    A.make_ge_than(0);
    A.make_le_than(15);

    B.make_ge_than(10);
    B.make_le_than(20);

    interval_templatet<int> expected;
    expected.make_ge_than(10);
    expected.make_le_than(15);

    A.meet(B);
    REQUIRE(A == expected);
  }

  SECTION("Intersect should... intersect 2")
  {
    // [0,5] intersec [10,20] = empty
    A.make_ge_than(0);
    A.make_le_than(5);

    B.make_ge_than(10);
    B.make_le_than(20);

    A.meet(B);
    REQUIRE(A.is_bottom());
  }

  SECTION("Add test 1")
  {
    // A: [1,10], B: [5,10], Result: [6,20]
    A.make_ge_than(1);
    A.make_le_than(10);

    B.make_ge_than(5);
    B.make_le_than(10);

    auto result = A + B;
    REQUIRE(result.lower_set);
    REQUIRE(result.upper_set);
    REQUIRE(result.lower == 6);
    REQUIRE(result.upper == 20);
  }

  SECTION("Add test 2")
  {
    // A: [-infinity,10], B: [5,10], Result: [-infinity,20]
    A.make_le_than(10);

    B.make_ge_than(5);
    B.make_le_than(10);

    auto result = A + B;
    REQUIRE_FALSE(result.lower_set);
    REQUIRE(result.upper_set);
    REQUIRE(result.upper == 20);
  }

  SECTION("Add test 3")
  {
    // A: [-infinity,10], B: [5,+infinity], Result: [-infinity,+infinity]
    A.make_le_than(10);

    B.make_ge_than(5);

    auto result = A + B;
    REQUIRE_FALSE(result.lower_set);
    REQUIRE_FALSE(result.upper_set);
  }

  SECTION("Sub test 1")
  {
    // A: [1,10], B: [5,10], Result: [-9,5]
    A.make_ge_than(1);
    A.make_le_than(10);

    B.make_ge_than(5);
    B.make_le_than(10);

    auto result = A - B;
    REQUIRE(result.lower_set);
    REQUIRE(result.upper_set);
    REQUIRE(result.lower == -9);
    REQUIRE(result.upper == 5);
  }

  SECTION("Sub test 2")
  {
    // A: [-infinity,10], B: [5,10], Result: [-infinity,5]
    A.make_le_than(10);

    B.make_ge_than(5);
    B.make_le_than(10);

    auto result = A - B;
    REQUIRE_FALSE(result.lower_set);
    REQUIRE(result.upper_set);
    REQUIRE(result.upper == 5);
  }

  SECTION("Sub test 3")
  {
    // A: [-infinity,10], B: [5,+infinity], Result: [-infinity,+infinity]
    A.make_le_than(10);

    B.make_ge_than(5);

    auto result = A - B;
    REQUIRE_FALSE(result.lower_set);
    REQUIRE(result.upper_set);
    REQUIRE(result.upper == 5);
  }

  SECTION("Mul test 1")
  {
    // A: [5,10], B: [-1,1], Result: [-10,+10]
    A.make_le_than(10);
    A.make_ge_than(5);
    B.make_ge_than(-1);
    B.make_le_than(1);

    auto result = A * B;
    REQUIRE(result.lower_set);
    REQUIRE(result.upper_set);
    REQUIRE(result.lower == -10);
    REQUIRE(result.upper == 10);
  }

  SECTION("Mul test 2")
  {
    // A: [-15,10], B: [-1,2], Result: [-30,+20]
    A.make_le_than(10);
    A.make_ge_than(-15);
    B.make_ge_than(-1);
    B.make_le_than(2);

    auto result = A * B;
    REQUIRE(result.lower_set);
    REQUIRE(result.upper_set);
    REQUIRE(result.lower == -30);
    REQUIRE(result.upper == 20);
  }

  SECTION("Div test 1")
  {
    // A: [4,10], B: [1,2], Result: [2,+10]
    A.make_le_than(10);
    A.make_ge_than(4);
    B.make_ge_than(1);
    B.make_le_than(2);

    auto result = A / B;
    REQUIRE(result.lower_set);
    REQUIRE(result.upper_set);
    REQUIRE(result.lower == 2);
    REQUIRE(result.upper == 10);
  }


  SECTION("Copy constructor")
  {
    // Just to be sure
    auto tmp_a = A;
    A.make_ge_than(0);
    REQUIRE(!tmp_a.lower_set);
    REQUIRE(A.lower_set);
  }
  SECTION("Contractor")
  {
    // From PRDC paper
    // X1: [0,20]
    // X2: [0, infinity]
    // X1 >= X2 ==> X2 <= X1 (which is what we are going to contract)
    // The result should be: A: [0,20] and B: [0,20]

    A.make_ge_than(0);
    A.make_le_than(20);
    B.make_ge_than(0);

    interval_templatet<int>::contract_interval_le(B, A);
    REQUIRE(A.lower_set);
    REQUIRE(A.upper_set);
    REQUIRE(A.lower == 0);
    REQUIRE(A.upper == 20);

    REQUIRE(B.lower_set);
    REQUIRE(B.upper_set);
    REQUIRE(B.lower == 0);
    REQUIRE(B.upper == 20);
  }
}

TEST_CASE("Wrapped Intervals tests", "[ai][interval-analysis]")
{
  unsigned N1 = 1;
  auto t1_unsigned = get_uint_type(N1);
  auto t1_signed = get_int_type(N1);

  unsigned N2 = 2;
  auto t2_unsigned = get_uint_type(N2);
  auto t2_signed = get_int_type(N2);

  SECTION("Upper bound test")
  {
    unsigned actual =
      wrapped_interval::get_upper_bound(t1_unsigned).to_uint64();
    unsigned expected = pow(2, N1 * 8);

    REQUIRE(actual == expected);
    REQUIRE(
      wrapped_interval::get_upper_bound(t1_unsigned) ==
      wrapped_interval::get_upper_bound(t1_signed));

    REQUIRE(
      wrapped_interval::get_upper_bound(t2_unsigned) ==
      wrapped_interval::get_upper_bound(t2_signed));

    REQUIRE(
      wrapped_interval::get_upper_bound(t1_unsigned) !=
      wrapped_interval::get_upper_bound(t2_signed));
  }

  SECTION("Wrapped less")
  {
    BigInt value1(10);
    BigInt value2(130);

    REQUIRE(wrapped_interval::wrapped_le(value1, 0, value2, t1_unsigned));
    REQUIRE(wrapped_interval::wrapped_le(value1, 0, value1, t1_unsigned));
    REQUIRE(!wrapped_interval::wrapped_le(value2, 0, value1, t1_unsigned));
  }

  SECTION("Init")
  {
    wrapped_interval A(t1_unsigned);
    wrapped_interval C(t2_unsigned);

    REQUIRE(A.lower == 0);
    REQUIRE(A.upper.to_uint64() == pow(2, N1 * 8));
    REQUIRE(!A.is_bottom());
    REQUIRE(A.is_top());

    REQUIRE(C.lower == 0);
    REQUIRE(C.upper.to_uint64() == pow(2, N2 * 8));
    REQUIRE(!C.is_bottom());
    REQUIRE(C.is_top());
  }

  SECTION("Non-overlap Interval")
  {
    wrapped_interval A(t1_unsigned);
    A.lower = 10;
    A.upper = 20;
    REQUIRE(A.contains(15));
    REQUIRE(!A.contains(150));
  }

  SECTION("Wrapping Interval")
  {
    wrapped_interval A(t1_unsigned);
    A.lower = 20;
    A.upper = 10;
    REQUIRE(!A.contains(15));
    REQUIRE(A.contains(25));
  }

  wrapped_interval A(t1_unsigned);
  wrapped_interval B(t1_unsigned);

  SECTION("Join/Meet when A is in B")
  {
    A.lower = 30;
    A.upper = 90;
    B.lower = 10;
    B.upper = 150;

    auto over_meet = wrapped_interval::over_meet(A, B);
    auto over_join = wrapped_interval::over_join(A, B);
    auto intersection = wrapped_interval::intersection(A, B);

    REQUIRE(over_meet.is_equal(A));
    REQUIRE(over_join.is_equal(B));
    REQUIRE(intersection.is_equal(A));
  }
  SECTION("Join/Meet when A do not overlap and B overlaps meets A in both ends")
  {
    A.lower = 10;
    A.upper = 190;
    B.lower = 150;
    B.upper = 20;
    // [10-190], [150-20]
    auto over_meet = wrapped_interval::over_meet(A, B);
    auto under_meet = wrapped_interval::under_meet(A, B);
    auto over_join = wrapped_interval::over_join(A, B);
    auto intersection = wrapped_interval::intersection(A, B);

    REQUIRE(over_join.is_top());
    REQUIRE(!under_meet.is_top());
    // Real intersection is: [150, 190] U [10, 20]
    wrapped_interval check(t1_unsigned);
    check.lower = 150;
    check.upper = 190;
    REQUIRE(check.is_included(intersection));
    check.lower = 10;
    check.upper = 20;
    REQUIRE(check.is_included(intersection));
  }
  SECTION("Join/Meet when A do not overlap and B overlaps meets A in one end")
  {
    A.lower = 10;
    A.upper = 150;
    B.lower = 200;
    B.upper = 100;
    // [10-150], [200-100]
    auto over_meet = wrapped_interval::over_meet(A, B);
    auto under_meet = wrapped_interval::under_meet(A, B);
    auto over_join = wrapped_interval::over_join(A, B);

    REQUIRE(!over_join.is_top());
    REQUIRE(over_join.lower == 200);
    REQUIRE(over_join.upper == 150);
    REQUIRE(under_meet == over_meet);
    REQUIRE(under_meet.lower == 10);
    REQUIRE(under_meet.upper == 100);

    // Intersection should be [10, 100]
    auto intersection = wrapped_interval::intersection(A, B);
    wrapped_interval check(t1_unsigned);
    check.lower = 10;
    check.upper = 100;
    REQUIRE(check.is_included(intersection));
  }

  SECTION("Join/Meet when A do not overlap and B overlaps and no intersection")
  {
    A.lower = 20;
    A.upper = 100;
    B.lower = 200;
    B.upper = 10;
    // [20-100], [200-10]
    auto over_meet = wrapped_interval::over_meet(A, B);
    auto under_meet = wrapped_interval::under_meet(A, B);
    auto over_join = wrapped_interval::over_join(A, B);

    REQUIRE(!over_join.is_top());
    REQUIRE(over_join.lower == 200);
    REQUIRE(over_join.upper == 100);

    // Intersection should be bottom
    auto intersection = wrapped_interval::intersection(A, B);
    REQUIRE(intersection.is_bottom());
  }
}
