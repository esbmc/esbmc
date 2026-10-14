// Accounting for the recursive-walk stack guard (issue #5048).
//
// The guard's job is to notice that a recursive tree walk has eaten its stack
// budget *before* the walk hits the guard page. Call sites respond by logging
// and aborting, so the interesting behaviour cannot be asserted through them;
// this drives the accounting directly with a budget small enough to trip in a
// handful of frames.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <util/base/stack_budget.h>

namespace
{
using walk_guardt = stack_budget_guardt<struct walk_tagt>;
using other_guardt = stack_budget_guardt<struct other_tagt>;

/** Recurse until the budget trips, returning the depth reached. `filler`
 *  keeps the frame from being optimised down to nothing. */
unsigned recurse(std::ptrdiff_t budget, unsigned limit, unsigned depth = 0)
{
  const walk_guardt guard;
  volatile char filler[512];
  filler[0] = static_cast<char>(depth);

  if (guard.exceeded(budget) || depth == limit)
    return depth;

  return recurse(budget, limit, depth + 1);
}
} // namespace

TEST_CASE("the outermost level has consumed nothing", "[util][stack]")
{
  const walk_guardt guard;
  REQUIRE(guard.bytes_used() == 0);
  // A budget of zero must not fire at depth 0, or every guarded walk would
  // refuse to start.
  REQUIRE_FALSE(guard.exceeded(0));
}

TEST_CASE("a nested level has consumed something", "[util][stack]")
{
  const walk_guardt outer;
  {
    const walk_guardt inner;
    REQUIRE(inner.bytes_used() > 0);
  }
  // Leaving the nested level restores the outermost measurement.
  REQUIRE(outer.bytes_used() == 0);
}

TEST_CASE("a small budget trips before the recursion limit", "[util][stack]")
{
  // 8 KiB against >=512-byte frames: tens of levels, far short of 100000.
  const unsigned depth = recurse(8 * 1024, 100000);
  REQUIRE(depth > 0);
  REQUIRE(depth < 100000);
}

TEST_CASE("a generous budget lets the walk finish", "[util][stack]")
{
  REQUIRE(recurse(default_stack_budget, 50) == 50);
}

TEST_CASE("depth resets once a walk unwinds", "[util][stack]")
{
  const unsigned first = recurse(8 * 1024, 100000);
  const unsigned second = recurse(8 * 1024, 100000);
  // A base pointer left behind by the first walk would make the second walk
  // measure from the wrong place and trip immediately.
  REQUIRE(second == first);
}

TEST_CASE(
  "walks with different tags do not measure each other",
  "[util][stack]")
{
  const other_guardt outer;
  const unsigned depth = recurse(8 * 1024, 100000);
  REQUIRE(depth > 0);
  REQUIRE(outer.bytes_used() == 0);
}
