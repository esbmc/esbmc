/*******************************************************************
 Module: Slicer dependency closure on the real symex_slicet

 H-A4's A4.2 obligation, discharged at Tier B (P0, I11/I12, M5) of
 docs/roadmap/goto-symex-verification-plan.md.

 The slicer is the only stage that *deletes* constraints (§2.3), so its one
 non-negotiable property is closure: a step it keeps must not read a definition
 it threw away. If it does, the retained step reads a free symbol instead of the
 value the program computed -- the solver is then answering a question about a
 different program, and the direction is unbounded.

 §7.1 specced this as a Tier-A transcription of `symex_slicet`. Per §6.4 it is
 here instead: the property is entirely observable on the equation the real
 slicer rewrites, so transcribing the algorithm would add drift risk (§9.1) and
 verify a copy. A4.1 (equisatisfiability) is a different matter, discharged
 empirically by H-C1 -- see §15 M5.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <set>
#include <string>
#include <vector>

#include <goto-symex/slice.h>

#include "ssa_validator.h"
#include "symex_run.h"

namespace
{
using symex_ssa::collect_ssa_reads;
using symex_ssa::is_ssa_symbol;

struct slice_result
{
  size_t retained;
  size_t ignored;
  /** Retained steps reading a name whose every definition was sliced away. */
  std::vector<std::string> dangling_reads;
  /** Assertions the slicer ignored -- it must never drop a claim. */
  size_t ignored_assertions;
  /** Retained reads of a retained definition: what closure is *about*. */
  size_t live_reads;
  /** Array stores neutralised in place. The slicer removes information two
   *  ways: `ignore` on the step, and rewriting an assignment's `cond` to
   *  `lhs == src` so a dead store encodes the identity. A closure check that
   *  reads only `ignore` is blind to the second -- deleting the
   *  `array_disqualified` consultation in slice.cpp leaves every `ignore` flag
   *  unchanged and so goes unnoticed. */
  size_t elided_stores;
};

/** True if the slicer rewrote this assignment's encoding away from its rhs. */
bool store_elided(const symex_target_equationt::SSA_stept &step)
{
  if (!step.is_assignment() || is_nil_expr(step.rhs) || is_nil_expr(step.cond))
    return false;
  if (!is_with2t(step.rhs))
    return false;
  return step.cond != expr2tc(equality2tc(step.lhs, step.rhs));
}

/** Run the real slicer over `eq` and audit what it left behind. */
slice_result slice_and_audit(symex_run::equation &run)
{
  symex_target_equationt &eq = run.get();

  // The equation symex hands back is unsliced: bmc.cpp runs the slicer
  // afterwards. Assert that, so this test cannot silently degrade into
  // re-slicing an already-sliced formula.
  for (const auto &step : eq.SSA_steps)
    REQUIRE_FALSE(step.ignore);

  symex_slicet slicer(run.options());
  slicer.run(eq.SSA_steps);

  std::set<std::string> retained_definitions;
  std::set<std::string> sliced_definitions;
  for (const auto &step : eq.SSA_steps)
  {
    if (!step.is_assignment() || !is_ssa_symbol(step.lhs))
      continue;
    const std::string name = to_symbol2t(step.lhs).get_symbol_name();
    (step.ignore ? sliced_definitions : retained_definitions).insert(name);
  }

  slice_result out{0, 0, {}, 0, 0, 0};
  for (const auto &step : eq.SSA_steps)
  {
    if (step.ignore)
    {
      out.ignored++;
      if (step.is_assert())
        out.ignored_assertions++;
      continue;
    }
    out.retained++;
    if (store_elided(step))
      out.elided_stores++;

    std::set<std::string> reads;
    collect_ssa_reads(step.guard, reads);
    if (!step.is_assignment())
      collect_ssa_reads(step.cond, reads);
    else if (store_elided(step))
      // An elided store keeps its rhs textually for trace construction,
      // but only the encoded condition reaches the formula. Collect
      // from that condition rather than reconstructing the expected
      // identity `lhs == src`, so a wrongly-encoded elided store still
      // surfaces as a closure violation instead of being assumed away.
      collect_ssa_reads(step.cond, reads);
    else
      collect_ssa_reads(step.rhs, reads);

    // A name with no definition anywhere is a free symbol (nondet, argument,
    // uninitialised global) and always was; only a name whose definitions were
    // *all* sliced is a closure violation.
    for (const std::string &name : reads)
    {
      if (retained_definitions.count(name))
        out.live_reads++;
      else if (sliced_definitions.count(name))
        out.dangling_reads.push_back(name);
    }
  }
  return out;
}

void require_closed(const std::string &src)
{
  symex_run::equation run(src);
  const slice_result r = slice_and_audit(run);

  // Non-vacuity: closure holds trivially if the slicer removed nothing, and
  // trivially if it removed everything.
  INFO("retained " << r.retained << ", ignored " << r.ignored);
  REQUIRE(r.ignored > 0);
  REQUIRE(r.retained > 0);
  // ... and the retained steps must actually read retained definitions, or the
  // closure check below has nothing to rule on.
  REQUIRE(r.live_reads > 0);

  INFO(
    "reads of sliced-away definitions: "
    << symex_ssa::describe(r.dangling_reads));
  CHECK(r.dangling_reads.empty());
  CHECK(r.ignored_assertions == 0);
}
} // namespace

TEST_CASE("the slicer keeps every definition it still reads", "[symex][slice]")
{
  require_closed(R"(
int nondet_int(void);
int main(void)
{
  int used = nondet_int();
  int dead = nondet_int();
  int chain = used + 1;
  dead = dead * 2;
  __ESBMC_assert(chain > used, "chain");
  return 0;
}
)");
}

TEST_CASE("closure survives a branch and a join", "[symex][slice]")
{
  require_closed(R"(
int nondet_int(void);
int main(void)
{
  int x = nondet_int();
  int unused = nondet_int();
  if (nondet_int() > 0)
    x = nondet_int();
  else
    unused = nondet_int();
  __ESBMC_assert(x != 424242, "phi read");
  return 0;
}
)");
}

TEST_CASE("closure survives a call and an unwound loop", "[symex][slice]")
{
  require_closed(R"(
int nondet_int(void);
int twice(int v) { return v + v; }
int main(void)
{
  int total = 0;
  int untouched = nondet_int();
  for (int i = 0; i < 3; i++)
    total += twice(nondet_int());
  __ESBMC_assert(total != 424242, "loop read");
  return 0;
}
)");
}

TEST_CASE("closure survives constant array indices", "[symex][slice]")
{
  // scan_array_uses / index_reads: a store to one constant index may be elided
  // only if no retained read can observe it.
  // The stored values are `nondet * 3` on purpose: bare symbol chains
  // now constant-propagate wholly and the read folds away, so the array
  // is never tracked and the stores die as plain dead code before the
  // elision. A multiplication over a nondet is refused by the
  // propagator, keeping the elision exercised.
  symex_run::equation run(R"(
int nondet_int(void);
int main(void)
{
  int arr[4];
  arr[0] = nondet_int() * 3;
  arr[1] = nondet_int() * 3;
  arr[2] = nondet_int() * 3;
  arr[3] = nondet_int() * 3;
  __ESBMC_assert(arr[1] != 424242, "read one index");
  return 0;
}
)");

  const slice_result r = slice_and_audit(run);
  INFO(
    "reads of sliced-away definitions: "
    << symex_ssa::describe(r.dangling_reads));
  CHECK(r.dangling_reads.empty());
  CHECK(r.ignored_assertions == 0);
  // Only index 1 is read, so the other three stores are dead. Requiring the
  // elision to happen keeps the symbolic-index case below honest: if the
  // optimisation stopped firing entirely, that case would pass for free.
  REQUIRE(r.elided_stores > 0);
}

TEST_CASE("a symbolic array index disqualifies the array", "[symex][slice]")
{
  // The shape H-A4's twin targets: with a symbolic index the slicer cannot know
  // which element is read, so it must retain every store to that array.
  symex_run::equation run(R"(
int nondet_int(void);
int main(void)
{
  int arr[4];
  arr[0] = nondet_int();
  arr[1] = nondet_int();
  arr[2] = nondet_int();
  arr[3] = nondet_int();
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 4);
  __ESBMC_assert(arr[i] != 424242, "symbolic read");
  return 0;
}
)");

  const slice_result r = slice_and_audit(run);
  INFO(
    "reads of sliced-away definitions: "
    << symex_ssa::describe(r.dangling_reads));
  CHECK(r.dangling_reads.empty());
  CHECK(r.ignored_assertions == 0);
  REQUIRE(r.live_reads > 0);
  // The read index is unknown, so no store to this array may be neutralised.
  CHECK(r.elided_stores == 0);
}
