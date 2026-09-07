#include <goto-programs/goto_invariant_synthesis.h>
#include <irep2/irep2_utils.h>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

using namespace invariant_synthesis;

namespace
{
type2tc u32()
{
  return get_uint32_type();
}

expr2tc var(const char *name)
{
  return symbol2tc(u32(), irep_idt(name));
}

expr2tc lit(int64_t v)
{
  return constant_int2tc(u32(), BigInt(v));
}
} // namespace

TEST_CASE("split_bound reads the counter and bound", "[invariant-synthesis]")
{
  expr2tc counter, bound;
  bool inclusive = false;

  SECTION("i < n is exclusive")
  {
    REQUIRE(
      split_bound(lessthan2tc(var("i"), var("n")), counter, bound, inclusive));
    REQUIRE(counter == var("i"));
    REQUIRE(bound == var("n"));
    REQUIRE_FALSE(inclusive);
  }

  SECTION("i <= n is inclusive")
  {
    REQUIRE(split_bound(
      lessthanequal2tc(var("i"), var("n")), counter, bound, inclusive));
    REQUIRE(counter == var("i"));
    REQUIRE(bound == var("n"));
    REQUIRE(inclusive);
  }

  SECTION("the sides are not commuted")
  {
    REQUIRE(
      split_bound(lessthan2tc(lit(3), var("i")), counter, bound, inclusive));
    REQUIRE(counter == lit(3));
    REQUIRE(bound == var("i"));
  }

  SECTION("comparisons the pass does not handle are declined")
  {
    REQUIRE_FALSE(split_bound(
      greaterthan2tc(var("i"), var("n")), counter, bound, inclusive));
    REQUIRE_FALSE(split_bound(
      greaterthanequal2tc(var("i"), var("n")), counter, bound, inclusive));
    REQUIRE_FALSE(
      split_bound(equality2tc(var("i"), var("n")), counter, bound, inclusive));
    REQUIRE_FALSE(
      split_bound(notequal2tc(var("i"), var("n")), counter, bound, inclusive));
    REQUIRE_FALSE(split_bound(var("i"), counter, bound, inclusive));
  }
}

TEST_CASE("is_self_increment matches lhs = lhs + e", "[invariant-synthesis]")
{
  expr2tc addend;

  SECTION("both operand orders are accepted")
  {
    REQUIRE(
      is_self_increment(var("s"), add2tc(u32(), var("s"), var("e")), addend));
    REQUIRE(addend == var("e"));

    REQUIRE(
      is_self_increment(var("s"), add2tc(u32(), var("e"), var("s")), addend));
    REQUIRE(addend == var("e"));
  }

  SECTION("s = s + s reports s as the addend")
  {
    REQUIRE(
      is_self_increment(var("s"), add2tc(u32(), var("s"), var("s")), addend));
    REQUIRE(addend == var("s"));
  }

  SECTION("a write to a different variable is not a self-increment")
  {
    REQUIRE_FALSE(
      is_self_increment(var("s"), add2tc(u32(), var("t"), var("e")), addend));
  }

  SECTION("only addition counts")
  {
    REQUIRE_FALSE(
      is_self_increment(var("s"), sub2tc(u32(), var("s"), var("e")), addend));
    REQUIRE_FALSE(
      is_self_increment(var("s"), mul2tc(u32(), var("s"), var("e")), addend));
    REQUIRE_FALSE(is_self_increment(var("s"), var("s"), addend));
    REQUIRE_FALSE(is_self_increment(var("s"), lit(1), addend));
  }
}

TEST_CASE(
  "entry_admits_two_disjunct_bound follows the case analysis",
  "[invariant-synthesis]")
{
  SECTION("i0 == 0 works for both comparisons")
  {
    REQUIRE(entry_admits_two_disjunct_bound(lit(0), true));
    REQUIRE(entry_admits_two_disjunct_bound(lit(0), false));
  }

  SECTION("i0 == 1 works only for <=, where E is B + 1")
  {
    REQUIRE(entry_admits_two_disjunct_bound(lit(1), true));
    REQUIRE_FALSE(entry_admits_two_disjunct_bound(lit(1), false));
  }

  SECTION("i0 >= 2 needs the third disjunct either way")
  {
    for (int64_t v : {2, 3, 64, 4096})
    {
      REQUIRE_FALSE(entry_admits_two_disjunct_bound(lit(v), true));
      REQUIRE_FALSE(entry_admits_two_disjunct_bound(lit(v), false));
    }
  }

  SECTION("a symbolic entry value is never admitted")
  {
    REQUIRE_FALSE(entry_admits_two_disjunct_bound(var("i0"), true));
    REQUIRE_FALSE(entry_admits_two_disjunct_bound(var("i0"), false));
    REQUIRE_FALSE(
      entry_admits_two_disjunct_bound(add2tc(u32(), var("i0"), lit(1)), true));
  }
}
