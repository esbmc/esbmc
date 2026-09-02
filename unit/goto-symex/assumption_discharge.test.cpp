/*******************************************************************
 Module: Discharging §7.3's assumption register on the real engine

 H-B7 of docs/roadmap/goto-symex-verification-plan.md.

 §7.3 records the assumptions the harnesses lean on but do not check. This file
 takes the one row that names H-B7 as its sole discharge and is still open: the
 slicer's dead-store elimination assumes every `with2t` store it elides has a
 `symbol2t` source and a constant index. The row asks H-B7 to "count the shapes
 reaching that branch", which is the useful form of the question -- the guard at
 `slice.cpp:249-254` makes the assumption true of whatever it lets through, so
 what matters is which shapes it *excludes*, and whether a shape that must never
 be elided could start qualifying.

 One such shape is a struct member store. `symex_assign` spells it
 `s' == s WITH ["f" := v]` with a `constant_string2t` field
 (`symex_assign.cpp:958-970`), while an array store carries a `constant_int2t`
 index, and only the latter passes the guard. This matters because the read-set
 the elision consults, `index_reads`, is populated exclusively from `index2t`
 reads (`slice.cpp:104-118`): a member read is a `member2t` and records nothing.
 A member store that reached the branch would therefore find its field "never
 read" and be dropped as dead -- a silent unsoundness, in the missed-bug
 direction, from a change no one would think of as touching the slicer.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <string>

#include <goto-symex/slice.h>

#include "symex_run.h"

namespace
{
struct shape_census
{
  /// `with2t` stores passing the slicer's array-elision guard.
  size_t qualifying = 0;
  /// Stores whose update field names a struct member rather than an index.
  size_t member_field = 0;
  /// Everything else: a symbolic index, or a source that is not a symbol.
  size_t other = 0;

  size_t total() const
  {
    return qualifying + member_field + other;
  }
};

/** The slicer's own guard, stated once (`slice.cpp:249-254`). */
bool qualifies_for_elision(const expr2tc &rhs)
{
  if (!is_with2t(rhs))
    return false;
  const with2t &with = to_with2t(rhs);
  return is_symbol2t(with.source_value) &&
         is_constant_int2t(with.update_field) &&
         !to_constant_int2t(with.update_field).value.is_negative();
}

shape_census census_of(const symex_target_equationt &eq)
{
  shape_census c;
  for (const auto &step : eq.SSA_steps)
  {
    if (!step.is_assignment() || is_nil_expr(step.rhs) || !is_with2t(step.rhs))
      continue;
    if (qualifies_for_elision(step.rhs))
      c.qualifying++;
    else if (is_constant_string2t(to_with2t(step.rhs).update_field))
      c.member_field++;
    else
      c.other++;
  }
  return c;
}

/** True if the slicer rewrote this store's encoding away from its rhs, i.e.
 *  elided it. Same predicate as `slice.test.cpp`'s `store_elided`. */
bool store_elided(const symex_target_equationt::SSA_stept &step)
{
  if (!step.is_assignment() || is_nil_expr(step.rhs) || is_nil_expr(step.cond))
    return false;
  if (!is_with2t(step.rhs))
    return false;
  return step.cond != expr2tc(equality2tc(step.lhs, step.rhs));
}

// Constant indices with one index read, so the other three stores are dead and
// the elision branch has to fire. Reading through an assertion, not the return
// value, keeps the array live: a returned expression is itself sliced away, and
// then the stores go with `ignore` rather than reaching the branch at all.
// The stored values are `nondet * 3`, not bare nondets, on purpose: a
// chain of (typecast) symbol stores now constant-propagates wholly, the
// read folds to the stored symbol at rename time, the array is never
// tracked, and every store dies as plain dead code before the elision
// branch can fire. A multiplication over a nondet is refused by the
// propagator, so these chains stay in the SSA and keep the exercise real.
const char *dead_array_store = R"(
int nondet_int(void);
int main(void)
{
  int a[4];
  a[0] = nondet_int() * 3;
  a[1] = nondet_int() * 3;
  a[2] = nondet_int() * 3;
  a[3] = nondet_int() * 3;
  __ESBMC_assert(a[1] != 424242, "read one index");
  return 0;
}
)";

// The same program indexed symbolically. No store qualifies, so none can be
// elided -- the incompleteness the guard buys in exchange for soundness.
const char *symbolic_index_store = R"(
int nondet_int(void);
int main(void)
{
  int a[4];
  int i = nondet_int() & 3;
  a[i] = nondet_int();
  a[1] = 7;
  return a[i];
}
)";

// Member stores, including one to a field that is never read. If these ever
// start qualifying, the never-read field is dropped and the assertion below
// becomes reachable on a program whose value it still depends on.
const char *member_store = R"(
int nondet_int(void);
struct s
{
  int read;
  int unread;
};
int main(void)
{
  struct s v;
  v.read = nondet_int();
  v.unread = nondet_int();
  return v.read;
}
)";
} // namespace

TEST_CASE(
  "every store the slicer elides has a symbol source and a constant index",
  "[symex][slice][assumptions]")
{
  symex_run::equation run(dead_array_store);
  symex_target_equationt &eq = run.get();

  symex_slicet slicer(run.options());
  slicer.run(eq.SSA_steps);

  size_t elided = 0;
  for (const auto &step : eq.SSA_steps)
    if (store_elided(step))
    {
      elided++;
      // §7.3's assumption, checked rather than assumed.
      REQUIRE(qualifies_for_elision(step.rhs));
      REQUIRE(is_symbol2t(step.lhs));
    }

  // Anti-vacuity: the assumption holds for want of a subject if nothing was
  // elided, and this program exists to make sure something is.
  REQUIRE(elided > 0);
}

TEST_CASE(
  "a struct member store never reaches the array-elision branch",
  "[symex][slice][assumptions]")
{
  symex_run::equation run(member_store);
  const shape_census c = census_of(run.get());

  INFO(
    "qualifying " << c.qualifying << ", member " << c.member_field << ", other "
                  << c.other);
  REQUIRE(c.member_field > 0);
  REQUIRE(c.qualifying == 0);
}

TEST_CASE(
  "a symbolic index disqualifies its store from elision",
  "[symex][slice][assumptions]")
{
  symex_run::equation run(symbolic_index_store);
  const shape_census c = census_of(run.get());

  INFO(
    "qualifying " << c.qualifying << ", member " << c.member_field << ", other "
                  << c.other);
  REQUIRE(c.total() > 0);
  REQUIRE(c.other > 0);
}

TEST_CASE(
  "constant-index stores do reach the branch",
  "[symex][slice][assumptions]")
{
  // The control for the two exclusions above: the guard is not simply always
  // false on the shapes symex produces.
  symex_run::equation run(dead_array_store);
  REQUIRE(census_of(run.get()).qualifying > 0);
}
