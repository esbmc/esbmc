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
    REQUIRE(result.lower);
    REQUIRE(result.upper);
    REQUIRE(*result.lower == 6);
    REQUIRE(*result.upper == 20);
  }

  SECTION("Add test 2")
  {
    // A: [-infinity,10], B: [5,10], Result: [-infinity,20]
    A.make_le_than(10);

    B.make_ge_than(5);
    B.make_le_than(10);

    auto result = A + B;
    REQUIRE_FALSE(result.lower);
    REQUIRE(result.upper);
    REQUIRE(*result.upper == 20);
  }

  SECTION("Add test 3")
  {
    // A: [-infinity,10], B: [5,+infinity], Result: [-infinity,+infinity]
    A.make_le_than(10);

    B.make_ge_than(5);

    auto result = A + B;
    REQUIRE_FALSE(result.lower);
    REQUIRE_FALSE(result.upper);
  }

  SECTION("Sub test 1")
  {
    // A: [1,10], B: [5,10], Result: [-9,5]
    A.make_ge_than(1);
    A.make_le_than(10);

    B.make_ge_than(5);
    B.make_le_than(10);

    auto result = A - B;
    REQUIRE(result.lower);
    REQUIRE(result.upper);
    REQUIRE(*result.lower == -9);
    REQUIRE(*result.upper == 5);
  }

  SECTION("Sub test 2")
  {
    // A: [-infinity,10], B: [5,10], Result: [-infinity,5]
    A.make_le_than(10);

    B.make_ge_than(5);
    B.make_le_than(10);

    auto result = A - B;
    REQUIRE_FALSE(result.lower);
    REQUIRE(result.upper);
    REQUIRE(*result.upper == 5);
  }

  SECTION("Sub test 3")
  {
    // A: [-infinity,10], B: [5,+infinity], Result: [-infinity,+infinity]
    A.make_le_than(10);

    B.make_ge_than(5);

    auto result = A - B;
    REQUIRE_FALSE(result.lower);
    REQUIRE(result.upper);
    REQUIRE(*result.upper == 5);
  }

  SECTION("Mul test 1")
  {
    // A: [5,10], B: [-1,1], Result: [-10,+10]
    A.make_le_than(10);
    A.make_ge_than(5);
    B.make_ge_than(-1);
    B.make_le_than(1);

    auto result = A * B;
    REQUIRE(result.lower);
    REQUIRE(result.upper);
    REQUIRE(*result.lower == -10);
    REQUIRE(*result.upper == 10);
  }

  SECTION("Mul test 2")
  {
    // A: [-15,10], B: [-1,2], Result: [-30,+20]
    A.make_le_than(10);
    A.make_ge_than(-15);
    B.make_ge_than(-1);
    B.make_le_than(2);

    auto result = A * B;
    REQUIRE(result.lower);
    REQUIRE(result.upper);
    REQUIRE(*result.lower == -30);
    REQUIRE(*result.upper == 20);
  }

  SECTION("Div test 1")
  {
    // A: [4,10], B: [1,2], Result: [2,+10]
    A.make_le_than(10);
    A.make_ge_than(4);
    B.make_ge_than(1);
    B.make_le_than(2);

    auto result = A / B;
    REQUIRE(result.lower);
    REQUIRE(result.upper);
    REQUIRE(*result.lower == 2);
    REQUIRE(*result.upper == 10);
  }

  SECTION("Copy constructor")
  {
    // Just to be sure
    auto tmp_a = A;
    A.make_ge_than(0);
    REQUIRE(!tmp_a.lower);
    REQUIRE(A.lower);
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
    REQUIRE(A.lower);
    REQUIRE(A.upper);
    REQUIRE(*A.lower == 0);
    REQUIRE(*A.upper == 20);

    REQUIRE(B.lower);
    REQUIRE(B.upper);
    REQUIRE(*B.lower == 0);
    REQUIRE(*B.upper == 20);
  }
}

TEST_CASE("Wrapped Intervals tests", "[ai][interval-analysis]")
{
  config.ansi_c.set_data_model(configt::ILP32);
  unsigned N1 = 8;
  auto t1_unsigned = get_uint_type(N1);
  auto t1_signed = get_int_type(N1);

  unsigned N2 = 16;
  auto t2_unsigned = get_uint_type(N2);
  auto t2_signed = get_int_type(N2);

  SECTION("Upper bound test")
  {
    unsigned actual =
      wrapped_interval(t1_unsigned).get_upper_bound().to_uint64();
    unsigned expected = pow(2, N1);

    REQUIRE(actual == expected);
    REQUIRE(
      wrapped_interval(t1_unsigned).get_upper_bound() ==
      wrapped_interval(t1_signed).get_upper_bound());

    REQUIRE(
      wrapped_interval(t2_unsigned).get_upper_bound() ==
      wrapped_interval(t2_signed).get_upper_bound());

    REQUIRE(
      wrapped_interval(t1_unsigned).get_upper_bound() !=
      wrapped_interval(t2_signed).get_upper_bound());
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

    REQUIRE(*A.lower == 0);
    REQUIRE((*A.upper).to_uint64() == pow(2, N1) - 1);
    REQUIRE(!A.is_bottom());
    CAPTURE(A.cardinality());
    REQUIRE(A.is_top());

    REQUIRE(*C.lower == 0);
    REQUIRE((*C.upper).to_uint64() == pow(2, N2) - 1);
    REQUIRE(!C.is_bottom());
    REQUIRE(C.is_top());

    // Char initialization
    for (int c = 0; c < 256; c++)
    {
      A.set_lower(c);
      A.set_upper(c);
      REQUIRE(*A.lower >= 0);
      REQUIRE(*A.upper >= 0);
      REQUIRE(A.get_lower() == c);
      REQUIRE(A.get_upper() == c);
    }
  }

  SECTION("Init Signed")
  {
    wrapped_interval A(t1_signed);
    CAPTURE((*A.lower).to_int64(), (*A.upper).to_uint64());
    REQUIRE(*A.lower == 0);
    REQUIRE((*A.upper).to_uint64() == pow(2, N1) - 1);
    REQUIRE(!A.is_bottom());
    REQUIRE(A.is_top());

    // Char initialization
    for (int c = -128; c < 128; c++)
    {
      A.set_lower(c);
      A.set_upper(c);
      REQUIRE(*A.lower >= 0);
      REQUIRE(*A.upper >= 0);
      REQUIRE(A.get_lower() == c);
      REQUIRE(A.get_upper() == c);
    }
  }

  SECTION("Relational Operators (Unsigned)")
  {
    wrapped_interval A(t1_unsigned);
    // [10, 250]
    A.set_lower(10);
    A.set_upper(250);
    A.make_le_than(50);
    REQUIRE(A.get_upper() == 50);
    REQUIRE(A.get_lower() == 10);

    // A: [250, 20]
    A.set_upper(20);
    A.set_lower(250);

    A.make_le_than(50); // [0, 20]

    REQUIRE(A.get_upper() == 20);
    A.make_ge_than(10);

    REQUIRE(A.get_lower() == 10);
  }

  SECTION("Relational Operators (Signed)")
  {
    wrapped_interval A(t1_signed);
    // A: [-128,127]

    REQUIRE(A.get_upper() == -1);
    REQUIRE(A.get_lower() == 0);

    A.make_le_than(500);
    A.make_le_than(50);
    // [-128,50]
    REQUIRE(A.get_upper() == 50);
    A.make_ge_than(-120);
    REQUIRE(A.get_lower() == -120);
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
  wrapped_interval As(t1_signed);
  wrapped_interval Bs(t1_signed);

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
    //REQUIRE(intersection.is_equal(A));
  }

  SECTION("Approx Union gets smallest gap")
  {
    A.set_upper(52);
    A.set_lower(52);

    B.set_lower(51);
    B.set_upper(51);

    B.approx_union_with(A);
    CAPTURE(B.lower, B.upper);
    REQUIRE(!B.is_bottom());
    REQUIRE(*B.lower == 51);
    REQUIRE(*B.upper == 52);
  }

  SECTION("Join/Meet from bug")
  {
    As.set_lower(-52);
    As.set_upper(-52);

    Bs.set_lower(51);
    Bs.set_upper(51);

    auto over_meet = wrapped_interval::over_meet(As, Bs);

    REQUIRE(over_meet.contains(-52));
    REQUIRE(over_meet.contains(51));
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

    // OVER_JOIN = [190, 150]
    CAPTURE(
      over_join.lower,
      over_join.upper,
      over_join.cardinality(),
      over_join.get_upper_bound(),
      over_join.is_bottom());
    REQUIRE(over_join.is_top());
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
    // Intersection should be [10, 100]
    auto intersection = wrapped_interval::intersection(A, B);
    wrapped_interval check(t1_unsigned);
    check.lower = 10;
    check.upper = 100;
    //REQUIRE(check.is_included(intersection));
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
    //REQUIRE(intersection.is_bottom());
  }

  SECTION("Addition Unsigned")
  {
    REQUIRE(*A.lower == 0);
    REQUIRE((*A.upper).to_uint64() == pow(2, N1) - 1);
    REQUIRE(!A.is_bottom());
    REQUIRE(A.is_top());

    // No Wrap
    A.set_lower(100);
    A.set_upper(150);

    B.set_lower(0);
    B.set_upper(1);

    auto C = A + B;
    REQUIRE(C.get_lower() == 100);
    REQUIRE(C.get_upper() == 151);

    // Wrap around
    A.set_lower(100);
    A.set_upper((long long)(pow(2, N1) - 1));

    B.set_lower(1);
    B.set_upper(2);

    C = A + B;
    REQUIRE(C.get_lower() == 101);
    REQUIRE(C.get_upper() == 1);
  }

  SECTION("Addition Unsigned wrap")
  {
    REQUIRE(A.lower == 0);
    REQUIRE((*A.upper).to_uint64() == pow(2, N1) - 1);
    REQUIRE(!A.is_bottom());
    REQUIRE(A.is_top());

    B.set_lower(0);
    B.set_upper(1);

    auto C = A + B;
    REQUIRE(C.is_top());
  }

  SECTION("Addition Signed")
  {
    REQUIRE(As.contains((long long)-pow(2, N1) / 2));
    REQUIRE(As.contains((long long)pow(2, N1) / 2 - 1));
    REQUIRE(!As.is_bottom());
    REQUIRE(As.is_top());

    // No Wrap
    As.set_lower(-100);
    As.set_upper(120);

    Bs.set_lower(-5);
    Bs.set_upper(5);

    auto C = As + Bs;
    CAPTURE(As.cardinality(), Bs.cardinality());
    REQUIRE(C.get_lower() == -105);
    REQUIRE(C.get_upper() == 125);

    // Wrap around
    As.set_lower(100);
    As.set_upper((long long)(pow(2, N1 - 1) - 1));

    Bs.set_lower(-1);
    Bs.set_upper(1);

    C = As + Bs;
    REQUIRE(C.get_lower() == 99);
    REQUIRE(C.get_upper().to_int64() == -pow(2, N1) / 2);
  }
}

TEST_CASE("Interval templates arithmetic operations", "[ai][interval-analysis]")
{
  config.ansi_c.set_data_model(configt::ILP32);
  unsigned N1 = 8;
  auto t1_unsigned = get_uint_type(N1);
  auto t1_signed = get_int_type(N1);

  SECTION("North Pole")
  {
    auto np = wrapped_interval::north_pole(t1_signed);
    REQUIRE((*np.lower == 127 && *np.upper == 128));
  }

  SECTION("South Pole")
  {
    auto np = wrapped_interval::south_pole(t1_signed);
    REQUIRE((*np.upper == 0 && *np.lower == 255));
  }

  SECTION("North Split")
  {
    wrapped_interval w1(t1_signed);
    w1.lower = 100;
    w1.upper = 120;

    auto w1_split = w1.nsplit();
    REQUIRE(w1_split.size() == 1);
    REQUIRE(w1_split[0] == w1);

    w1.lower = 100;
    w1.upper = 150;
    w1_split = w1.nsplit();
    REQUIRE(w1_split.size() == 2);

    // The right part is returned as first element! (not a rule though)
    REQUIRE(*w1_split[0].lower == 128);
    REQUIRE(*w1_split[0].upper == 150);
    REQUIRE(*w1_split[1].lower == 100);
    REQUIRE(*w1_split[1].upper == 127);
  }

  SECTION("South Split")
  {
    wrapped_interval w1(t1_signed);
    w1.lower = 100;
    w1.upper = 250;

    auto w1_split = w1.ssplit();
    REQUIRE(w1_split.size() == 1);
    REQUIRE(w1_split[0] == w1);

    w1.lower = 200;
    w1.upper = 150;
    w1_split = w1.ssplit();
    REQUIRE(w1.contains(0));
    REQUIRE(w1.contains(255));
    REQUIRE(w1_split.size() == 2);

    // The left part is returned as first element! (not a rule though)
    REQUIRE(*w1_split[0].lower == 0);
    REQUIRE(*w1_split[0].upper == 150);
    REQUIRE(*w1_split[1].lower == 200);
    REQUIRE(*w1_split[1].upper == 255);
  }

  SECTION("Cut")
  {
    wrapped_interval w(t1_signed);
    w.lower = 255;
    w.upper = 129;

    auto cut = wrapped_interval::cut(w);

    bool check1 = false;
    bool check2 = false;
    bool check3 = false;

    wrapped_interval w1(t1_signed), w2(t1_signed), w3(t1_signed);
    w1.lower = 255;
    w1.upper = 255;

    w2.lower = 0;
    w2.upper = 127;

    w3.lower = 128;
    w3.upper = 129;

    // There is no way for me to check the order
    for (auto &c : cut)
    {
      if (c == w1)
        check1 = true;
      if (c == w2)
        check2 = true;
      if (c == w3)
        check3 = true;
    }

    REQUIRE(check1);
    REQUIRE(check2);
    REQUIRE(check3);
  }

  SECTION("MSB")
  {
    wrapped_interval w(t1_signed);
    w.lower = 255;
    w.upper = 127;

    REQUIRE(w.most_significant_bit(*w.lower));
    REQUIRE(!w.most_significant_bit(*w.upper));

    w.lower = 25;
    w.upper = 130;

    REQUIRE(!w.most_significant_bit(*w.lower));
    REQUIRE(w.most_significant_bit(*w.upper));
  }

  SECTION("Difference")
  {
    wrapped_interval w1(t1_signed);
    w1.lower = 100;
    w1.upper = 120;

    wrapped_interval w2(t1_signed);
    w2.lower = 90;
    w2.upper = 110;

    auto result = w1.difference(w1, w2);
    CAPTURE(result.lower, result.upper);
    REQUIRE(*result.lower == 111);
    REQUIRE(*result.upper == 120);
  }
}

TEST_CASE("Interval templates multiplication", "[ai][interval-analysis]")
{
  config.ansi_c.set_data_model(configt::ILP32);
  unsigned N1 = 8;
  auto t1_unsigned = get_uint_type(N1);
  auto t1_signed = get_int_type(N1);

  SECTION("Multiply unsigned")
  {
    wrapped_interval w1(t1_unsigned);
    w1.lower = 5;
    w1.upper = 10;

    wrapped_interval w2(t1_unsigned);
    w2.lower = 2;
    w2.upper = 10;

    auto w3 = w1 * w2;

    CAPTURE(w3.lower, w3.upper);
    REQUIRE(*w3.lower == 10);
    REQUIRE(*w3.upper == 100);

    w1.lower = 1;
    w1.upper = 2;

    w2.lower = 1;
    w2.upper = 250;

    w3 = w1 * w2;
    CAPTURE(w3.lower, w3.upper);
    REQUIRE(*w3.lower == 0);
    REQUIRE(*w3.upper == 254);
  }

  SECTION("Multiply signed")
  {
    wrapped_interval w1(t1_signed);
    w1.lower = 1;
    w1.upper = 10;

    wrapped_interval w2(t1_signed);
    w2.lower = 255;
    w2.upper = 255;

    auto w3 = w1 * w2;

    CAPTURE(w3.lower, w3.upper);
    REQUIRE(*w3.lower == 246);
    REQUIRE(*w3.upper == 255);
  }
}

TEST_CASE("Interval templates division", "[ai][interval-analysis]")
{
  config.ansi_c.set_data_model(configt::ILP32);
  unsigned N1 = 8;
  auto t1_unsigned = get_uint_type(N1);
  auto t1_signed = get_int_type(N1);

  SECTION("Division unsigned")
  {
    wrapped_interval w1(t1_unsigned);
    w1.lower = 5;
    w1.upper = 20;

    wrapped_interval w2(t1_unsigned);
    w2.lower = 2;
    w2.upper = 10;

    auto w3 = w1 / w2;

    CAPTURE(w3.lower, w3.upper);
    REQUIRE(*w3.lower == 0);
    REQUIRE(*w3.upper == 10);

    w1.lower = 10;
    w1.upper = 20;

    w2.lower = 2;
    w2.upper = 10;

    w3 = w1 / w2;
    CAPTURE(w3.lower, w3.upper);
    REQUIRE(*w3.lower == 1);
    REQUIRE(*w3.upper == 10);
  }

  SECTION("Division signed")
  {
    wrapped_interval w1(t1_signed);
    w1.lower = 10;
    w1.upper = 20; // 10, 20

    wrapped_interval w2(t1_signed);
    w2.lower = 254; // -2
    w2.upper = 255; // -1
    // -5 (251), -20: (236)
    auto w3 = w1 / w2;

    CAPTURE(w3.lower, w3.upper);
    REQUIRE(*w3.lower == 236);
    REQUIRE(*w3.upper == 251);
  }
}

TEST_CASE("Wrapped Interval Typecast", "[ai][interval-analysis]")
{
  config.ansi_c.set_data_model(configt::ILP32);
  unsigned N1 = 8, N2 = 16;
  auto t1_unsigned = get_uint_type(N1);
  auto t1_signed = get_int_type(N1);
  auto t2_unsigned = get_uint_type(N2);
  auto t2_signed = get_int_type(N2);

  SECTION("Truncation (unsigned)")
  {
    wrapped_interval w1(t2_unsigned);
    w1.lower = 30;
    w1.upper = 250;

    auto result1 = w1.trunc(t1_unsigned);
    CAPTURE(result1.lower, result1.upper);
    REQUIRE(result1.lower == 30);
    REQUIRE(result1.upper == 250);

    w1.lower = 256;
    w1.upper = 260;

    result1 = w1.trunc(t1_unsigned);
    CAPTURE(result1.lower, result1.upper);
    REQUIRE(*result1.lower == 0);
    REQUIRE(*result1.upper == 4);
  }

  SECTION("Truncation (signed)")
  {
    wrapped_interval w1(t2_signed);
    w1.lower = 30;
    w1.upper = 128;

    auto result1 = w1.trunc(t1_signed);
    CAPTURE(result1.lower, result1.upper);
    REQUIRE(*result1.lower == 30);
    REQUIRE(*result1.upper == 128);

    w1.lower = 30;
    w1.upper = 257;

    result1 = w1.trunc(t1_signed);
    CAPTURE(result1.lower, result1.upper);
    REQUIRE(*result1.lower == 30);
    REQUIRE(*result1.upper == 1);
  }

  SECTION("Unsigned to unsigned")
  {
    wrapped_interval w1(t1_unsigned);
    w1.lower = 255;
    w1.upper = 255;

    auto result1 = wrapped_interval::cast(w1, t2_unsigned);
    CAPTURE(result1.lower, result1.upper);
    REQUIRE(*result1.upper == 255);
  }

  SECTION("Signed to Signed 1")
  {
    wrapped_interval w1(t1_signed);
    w1.lower = 0xff;
    w1.upper = 0xff;

    auto result1 = wrapped_interval::cast(w1, t2_signed);
    CAPTURE(result1.lower, result1.upper);
    REQUIRE(*result1.upper == 0xffff);
  }

  SECTION("Signed to Signed 2")
  {
    wrapped_interval w1(t1_signed);
    w1.lower = 20;
    w1.upper = 20;

    auto result1 = wrapped_interval::cast(w1, t2_signed);
    CAPTURE(result1.lower, result1.upper);
    REQUIRE(*result1.upper == 20);
  }

  SECTION("Unsigned to signed")
  {
    wrapped_interval w1(t1_unsigned);
    w1.lower = 255;
    w1.upper = 255;

    auto result1 = wrapped_interval::cast(w1, t2_signed);
    CAPTURE(result1.lower, result1.upper);
    REQUIRE(*result1.upper == 255);
  }

  SECTION("Signed to Unsigned")
  {
    wrapped_interval w1(t1_signed);
    w1.lower = 0xff;
    w1.upper = 0xff;

    auto result1 = wrapped_interval::cast(w1, t2_unsigned);
    CAPTURE(result1.lower, result1.upper);
    REQUIRE(*result1.upper == 0xffff);
  }

  SECTION("Unsigned to BOOL")
  {
    wrapped_interval w1(t1_unsigned);

    w1.lower = 0xff;
    w1.upper = 0xff;

    auto result1 = wrapped_interval::cast(w1, get_bool_type());
    CAPTURE(result1.lower, 1);
    REQUIRE(*result1.upper == 1);
  }

  SECTION("Unsigned to BOOL 2")
  {
    wrapped_interval w1(t1_unsigned);

    w1.lower = 0;
    w1.upper = 0;

    auto result1 = wrapped_interval::cast(w1, get_bool_type());
    CAPTURE(result1.lower, 0);
    REQUIRE(*result1.upper == 0);
  }

  SECTION("Unsigned to BOOL 3")
  {
    wrapped_interval w1(t1_unsigned);

    w1.lower = 0;
    w1.upper = 254;

    auto result1 = wrapped_interval::cast(w1, get_bool_type());
    CAPTURE(result1.lower, 0);
    REQUIRE(*result1.upper == 1);
  }
}

TEST_CASE("Wrapped Interval Left Shift", "[ai][interval-analysis]")
{
  config.ansi_c.set_data_model(configt::ILP32);
  unsigned N1 = 8;
  auto t1_unsigned = get_uint_type(N1);
  auto t1_signed = get_int_type(N1);
  SECTION("Left shift")
  {
    wrapped_interval w(t1_unsigned);
    w.lower = 30;
    w.upper = 80;

    auto result1 = w.left_shift(1);
    CAPTURE(result1.lower, result1.upper);
    REQUIRE(*result1.lower == 60);
    REQUIRE(*result1.upper == 160);

    w.lower = 50;
    w.upper = 100;

    auto result2 = w.left_shift(3);
    CAPTURE(result2.lower, result2.upper);
    REQUIRE(*result2.lower == 0);
    REQUIRE(*result2.upper == 248);
  }

  SECTION("Left shift signed")
  {
    wrapped_interval w(t1_signed);
    w.lower = 252; // -4
    w.upper = 254; // -2

    auto result1 = w.left_shift(1);
    CAPTURE(result1.lower, result1.upper);
    REQUIRE(*result1.lower == 0);
    REQUIRE(*result1.upper == 254);
  }

  SECTION("Logical right shift")
  {
    wrapped_interval w(t1_unsigned);
    w.lower = 30;
    w.upper = 80;

    auto result1 = w.logical_right_shift(1);
    CAPTURE(result1.lower, result1.upper);
    REQUIRE(*result1.lower == 15);
    REQUIRE(*result1.upper == 40);

    w.lower = 250;
    w.upper = 20; // [0, 127]

    auto result2 = w.logical_right_shift(1);
    CAPTURE(result2.lower, result2.upper);
    REQUIRE(*result2.lower == 0);
    REQUIRE(*result2.upper == 127);
  }

  SECTION("Arithmetic right shift")
  {
    wrapped_interval w(t1_signed);
    w.lower = 30;
    w.upper = 80;

    auto result1 = w.arithmetic_right_shift(1);
    CAPTURE(result1.lower, result1.upper);
    REQUIRE(result1.lower == 15);
    REQUIRE(result1.upper == 40);

    w.lower = 252;
    w.upper = 254;

    auto result2 = w.arithmetic_right_shift(1);
    CAPTURE(result2.lower, result2.upper);
    REQUIRE(*result2.lower == 254);
    REQUIRE(*result2.upper == 255);
  }
}

TEST_CASE("Remainder Operations", "[ai][interval-analysis]")
{
  config.ansi_c.set_data_model(configt::ILP32);
  unsigned N1 = 8;
  auto t1_unsigned = get_uint_type(N1);
  auto t1_signed = get_int_type(N1);

  SECTION("Singletons")
  {
    wrapped_interval w1(t1_unsigned);
    wrapped_interval w2(t1_unsigned);

    w1.lower = 10;
    w1.upper = 10;

    w2.lower = 10;
    w2.upper = 10;

    auto result1 = w1 % w2;
    REQUIRE(*result1.lower == 0);
    REQUIRE(*result1.upper == 0);

    w2.lower = 9;
    w2.upper = 9;

    auto result2 = w1 % w2;
    REQUIRE(*result2.lower == 1);
    REQUIRE(*result2.upper == 1);

    w2.lower = 11;
    w2.upper = 11;

    auto result3 = w1 % w2;
    REQUIRE(*result3.lower == 10);
    REQUIRE(*result3.upper == 10);

    wrapped_interval w3(t1_signed);
    wrapped_interval w4(t1_signed);

    w3.lower = 10;
    w3.upper = 10;

    w4.lower = 10;
    w4.upper = 10;

    auto result4 = w3 % w4;
    REQUIRE(*result4.lower == 0);
    REQUIRE(*result4.upper == 0);

    w4.lower = 9;
    w4.upper = 9;

    auto result5 = w3 % w4;
    REQUIRE(*result5.lower == 1);
    REQUIRE(*result5.upper == 1);

    w4.lower = 11;
    w4.upper = 11;

    auto result6 = w3 % w4;
    REQUIRE(*result6.lower == 10);
    REQUIRE(*result6.upper == 10);

    w3.set_lower(-10);
    w3.set_upper(-10);

    w4.lower = 10;
    w4.upper = 10;

    auto result7 = w3 % w4;
    REQUIRE(*result7.lower == 0);
    REQUIRE(*result7.upper == 0);

    w4.lower = 9;
    w4.upper = 9;

    auto result8 = w3 % w4;
    REQUIRE(result8.get_lower() == -1);
    REQUIRE(result8.get_lower() == -1);

    w4.lower = 11;
    w4.upper = 11;

    auto result9 = w3 % w4;
    REQUIRE(result9.get_lower() == -10);
    REQUIRE(result9.get_lower() == -10);
  }

  SECTION("Non singletons (unsigned)")
  {
    wrapped_interval w1(t1_unsigned);
    wrapped_interval w2(t1_unsigned);

    w1.lower = 9;
    w1.upper = 10;

    w2.lower = 5;
    w2.upper = 5;

    auto result1 = w1 % w2;
    REQUIRE(*result1.lower == 0);
    REQUIRE(*result1.upper == 4);

    w1.lower = 10;
    w1.upper = 8;

    auto result2 = w1 % w2;
    REQUIRE(*result2.lower == 0);
    REQUIRE(*result2.upper == 4);

    w1.lower = 10;
    w1.upper = 10;

    w2.lower = 4;
    w2.upper = 5;

    auto result3 = w1 % w2;
    REQUIRE(*result3.lower == 0);
    REQUIRE(*result3.upper == 2);

    w2.lower = 5;
    w2.upper = 1;

    // TODO: we can probably optimize this!
    auto result4 = w1 % w2;
    REQUIRE(*result4.lower == 0);
    REQUIRE(*result4.upper == 254);
  }

  SECTION("Non singletons (signed)")
  {
    wrapped_interval w1(t1_signed);
    wrapped_interval w2(t1_signed);

    w1.lower = 9;
    w1.upper = 10;

    w2.lower = 5;
    w2.upper = 5;

    auto result1 = w1 % w2;
    REQUIRE(*result1.lower == 0);
    REQUIRE(*result1.upper == 4);

    w1.lower = 10;
    w1.upper = 8;

    auto result2 = w1 % w2;
    CAPTURE(result2.upper, result2.lower);
    REQUIRE(*result2.lower == 252); // -4
    REQUIRE(*result2.upper == 4);   // 4

    w1.lower = 200;
    w1.upper = 250;

    w2.lower = 4; // [-3,0]
    w2.upper = 4;

    auto result3 = w1 % w2;
    REQUIRE(*result3.lower == 253);
    REQUIRE(*result3.upper == 0);
  }
}

TEST_CASE("Bitor Operations", "[ai][interval-analysis]")
{
  config.ansi_c.set_data_model(configt::ILP32);
  unsigned N1 = 8;
  auto t1_unsigned = get_uint_type(N1);

  wrapped_interval w1(t1_unsigned);
  wrapped_interval w2(t1_unsigned);
  SECTION("Singletons")
  {
    w1.lower = 10;  // 0x0A
    w1.upper = 10;  // 0x0A
    w2.lower = 160; // 0xA0
    w2.upper = 160; // 0xA0

    auto result = w1 | w2;
    CAPTURE(result.lower, result.upper);
    REQUIRE(*result.lower == 170); // 0xAA
    REQUIRE(*result.upper == 170); // 0xAA
  }

  SECTION("Intervals")
  {
    w1.lower = 10;  // 0x0A
    w1.upper = 12;  // 0x0C
    w2.lower = 161; // 0xA1
    w2.upper = 162; // 0xA2

    auto result = w1 | w2;
    CAPTURE(result.lower, result.upper);
    REQUIRE(*result.lower == 170);
    REQUIRE(*result.upper == 174);
  }
  /*
  SECTION("Intervals Overlap")
  {
    w1.lower = 250; // 0xFA
    w1.upper = 2; // 0x02
    w2.lower = 1; // 0x01
    w2.upper = 1; // 0x01

    auto result = w1 | w2;
    CAPTURE(result.lower, result.upper);
    REQUIRE(result.lower == 251);
    REQUIRE(result.upper == 3);
  }
   */
}

TEST_CASE("Bitand Operations", "[ai][interval-analysis]")
{
  config.ansi_c.set_data_model(configt::ILP32);
  unsigned N1 = 32;
  auto t1_unsigned = get_uint_type(N1);

  uint64_t m = (uint64_t)1 << 31;
  REQUIRE(m == 0x80000000);

  wrapped_interval w1(t1_unsigned);
  wrapped_interval w2(t1_unsigned);
  SECTION("Singletons disjunct")
  {
    w1.lower = 10;  // 0x0A
    w1.upper = 10;  // 0x0A
    w2.lower = 160; // 0xA0
    w2.upper = 160; // 0xA0

    auto result = w1 & w2;
    CAPTURE(result.lower, result.upper);
    REQUIRE(*result.lower == 0); // 0x00
    REQUIRE(*result.upper == 0); // 0x00
  }

  SECTION("Singletons")
  {
    w1.lower = 11;  // 0x0B
    w1.upper = 11;  // 0x0B
    w2.lower = 161; // 0xA1
    w2.upper = 161; // 0xA1

    auto result = w1 & w2;
    CAPTURE(result.lower, result.upper);
    REQUIRE(*result.lower == 1); // 0x01
    REQUIRE(*result.upper == 1); // 0x01
  }

  SECTION("Intervals")
  {
    w1.lower = 10;  // 0x0A
    w1.upper = 12;  // 0x0C
    w2.lower = 161; // 0xA1
    w2.upper = 162; // 0xA2

    auto result = w1 & w2;
    CAPTURE(result.lower, result.upper);
    REQUIRE(*result.lower == 0);
    REQUIRE(*result.upper == 2);
  }

  SECTION("Hardware issue")
  {
    w1.lower = 0; // 0x00
    w1.upper = 1; // 0x01
    w2.lower = 1; // 0x01
    w2.upper = 1; // 0x01

    auto result = w1 & w2;
    CAPTURE(result.lower, result.upper);
    REQUIRE(*result.lower == 0);
    REQUIRE(*result.upper == 1);
  }

  SECTION("Hardware issue 2")
  {
    auto t2_unsigned = get_uint_type(64);
    auto w_ulong(t2_unsigned);
    w2.lower = 1; // 0x01
    w2.upper = 1; // 0x01

    auto result = w1 & w2;
    CAPTURE(result.lower, result.upper);
    REQUIRE(*result.lower == 0);
    REQUIRE(*result.upper == 1);
  }
}

TEST_CASE("Bitxor Operations", "[ai][interval-analysis]")
{
  config.ansi_c.set_data_model(configt::ILP32);
  unsigned N1 = 8;
  auto t1_unsigned = get_uint_type(N1);

  wrapped_interval w1(t1_unsigned);
  wrapped_interval w2(t1_unsigned);
  SECTION("Singletons disjunct")
  {
    w1.lower = 10;  // 0x0A
    w1.upper = 10;  // 0x0A
    w2.lower = 160; // 0xA0
    w2.upper = 160; // 0xA0

    auto result = w1 ^ w2;
    CAPTURE(result.lower, result.upper);
    REQUIRE(*result.lower == 170); // 0xAA
    REQUIRE(*result.upper == 170); // 0xAA
  }

  SECTION("Singletons")
  {
    w1.lower = 1;   // 0x01
    w1.upper = 1;   // 0x01
    w2.lower = 161; // 0xA1
    w2.upper = 161; // 0xA1

    auto result = w1 ^ w2;
    CAPTURE(result.lower, result.upper);
    REQUIRE(*result.lower == 160); // 0xA0
    REQUIRE(*result.upper == 160); // 0xA0
  }

  SECTION("Intervals")
  {
    w1.lower = 10;  // 0x0A   0000 1010
    w1.upper = 12;  // 0x0C   0000 1100
    w2.lower = 161; // 0xA1  1011 0001
    w2.upper = 162; // 0xA2  1011 0010

    auto result = w1 ^ w2;
    CAPTURE(result.lower, result.upper);
    REQUIRE(*result.lower == 168); // 1010 1100
    REQUIRE(*result.upper == 174); // 1010 1110
  }
}

TEST_CASE("Bitnot Operations", "[ai][interval-analysis]")
{
  config.ansi_c.set_data_model(configt::ILP32);
  unsigned N1 = 8;
  auto t1_unsigned = get_uint_type(N1);
  wrapped_interval w(t1_unsigned);
  SECTION("Singletons")
  {
    w.lower = 1; // 0x01
    w.upper = 1; // 0x01

    auto r = wrapped_interval::bitnot(w);
    CAPTURE(r.lower, r.upper);
    REQUIRE(*r.lower == 254);
    REQUIRE(*r.upper == 254);

    w.lower = 250; // 0xFA
    w.upper = 250; // 0xFA

    r = wrapped_interval::bitnot(w);
    CAPTURE(r.lower, r.upper);
    REQUIRE(*r.lower == 5); // 0x05
    REQUIRE(*r.upper == 5); // 0x05
  }

  SECTION("Intervals")
  {
    w.lower = 0; // 0x00
    w.upper = 1; // 0x01

    auto r = wrapped_interval::bitnot(w);
    CAPTURE(r.lower, r.upper);
    REQUIRE(*r.lower == 254); // 0xFE
    REQUIRE(*r.upper == 255); // 0xFF
  }
}

TEST_CASE("Wrapped interval bounds", "[ai][interval-analysis]")
{
  config.ansi_c.set_data_model(configt::ILP32);
  wrapped_interval w1(get_int_type(8));
  std::pair<BigInt, BigInt> result;
  SECTION("[10, 127] --> <10, 127>")
  {
    w1.lower = 10;
    w1.upper = 127;

    result = w1.get_interval_bounds();
    CAPTURE(result.first, result.second);
    REQUIRE(result.first == 10);
    REQUIRE(result.second == 127);
  }

  SECTION("[10, 128] --> <-128, 127>")
  {
    w1.lower = 10;
    w1.upper = 128;

    result = w1.get_interval_bounds();
    CAPTURE(result.first, result.second);
    REQUIRE(result.first == -128);
    REQUIRE(result.second == 127);
  }

  SECTION("[10, 255] --> <-128, 127>")
  {
    w1.lower = 10;
    w1.upper = 255;

    result = w1.get_interval_bounds();
    CAPTURE(result.first, result.second);
    REQUIRE(result.first == -128);
    REQUIRE(result.second == 127);
  }

  SECTION("[129, 130] --> <-127, -126>")
  {
    w1.lower = 129;
    w1.upper = 130;

    result = w1.get_interval_bounds();
    CAPTURE(result.first, result.second);
    REQUIRE(result.first == -127);
    REQUIRE(result.second == -126);
  }

  SECTION("[255, 10] --> <-1, 10>")
  {
    w1.lower = 255;
    w1.upper = 10;

    result = w1.get_interval_bounds();
    CAPTURE(result.first, result.second);
    REQUIRE(result.first == -1);
    REQUIRE(result.second == 10);
  }
}
