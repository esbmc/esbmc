/*******************************************************************
 Module: Run-to-run determinism of the produced equation

 H-B2 of docs/roadmap/goto-symex-verification-plan.md (P10, M4).

 Objective 7: two symex runs over the same program in the same configuration
 must produce the same equation. This one is load-bearing for the rest of the
 plan rather than for any single verdict — a reordering invisible in any one run
 makes every other result here unreproducible and silently invalidates
 regression pinning.

 The run fixture lives in symex_run.h: each equation refers to its own context
 by reference, so comparing two runs means keeping two bundles alive.

 The property holds strictly. It did not always: two counters naming symex
 objects were `static thread_local` and reset nowhere, so a second run in the
 same process numbered its objects from where the first stopped (R15, fixed by
 resetting both in `setup_for_new_explore`). The canonicalising comparator is
 kept because it localises a failure — a diff that survives normalisation is a
 different defect from one that does not — but the heap case now asserts strict
 equality, which is what R15's fix bought.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <irep2/irep2_expr.h>

#include "symex_run.h"

namespace
{
size_t crc_of(const expr2tc &e)
{
  return is_nil_expr(e) ? 0 : e.crc();
}

struct step_crc
{
  unsigned type;
  bool ignore;
  bool hidden;
  size_t guard;
  size_t lhs;
  size_t rhs;
  size_t cond;

  bool operator==(const step_crc &) const = default;
};

std::vector<step_crc> crcs(const symex_target_equationt &eq)
{
  std::vector<step_crc> steps;
  steps.reserve(eq.SSA_steps.size());
  for (const auto &s : eq.SSA_steps)
    steps.push_back(
      {static_cast<unsigned>(s.type),
       s.ignore,
       s.hidden,
       crc_of(s.guard),
       crc_of(s.lhs),
       crc_of(s.rhs),
       crc_of(s.cond)});
  return steps;
}

/** The same steps with R15's two counters normalised away. Text rather than a
 *  crc because the numbering sits inside symbol names, which a structural hash
 *  cannot reach without rewriting the expressions. */
std::vector<std::string> canonical(const symex_target_equationt &eq)
{
  static const std::regex object_id("(dynamic_|symex::invalid_object)[0-9]+");
  std::vector<std::string> steps;
  steps.reserve(eq.SSA_steps.size());
  for (const auto &s : eq.SSA_steps)
  {
    std::ostringstream text;
    text << s.type << '/' << s.ignore << '/' << s.hidden;
    for (const expr2tc *e : {&s.guard, &s.lhs, &s.rhs, &s.cond})
      text << '|' << (is_nil_expr(*e) ? std::string("-") : (*e)->pretty(0));
    steps.push_back(std::regex_replace(text.str(), object_id, "$1N"));
  }
  return steps;
}

/** Report the first divergent step rather than only "unequal": for a 50-step
 *  equation the latter is not a usable diagnostic. */
template <typename T>
void require_equal_steps(const std::vector<T> &a, const std::vector<T> &b)
{
  // Non-vacuity: an empty equation would make any two runs trivially equal.
  REQUIRE(a.size() > 0);
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); i++)
    if (!(a[i] == b[i]))
      FAIL("equations diverge at step " << i << " of " << a.size());
}

void require_identical(const std::string &src)
{
  symex_run::equation first(src), second(src);
  require_equal_steps(crcs(first.get()), crcs(second.get()));
}

void require_identical_modulo_object_ids(const std::string &src)
{
  symex_run::equation first(src), second(src);
  require_equal_steps(canonical(first.get()), canonical(second.get()));
}
} // namespace

TEST_CASE("the comparators distinguish two programs", "[symex][determinism]")
{
  // The control for every case below: a comparator that cannot tell two
  // equations apart would pass all of them without checking anything.
  symex_run::equation one("int main(void) { int x = 1; return x; }");
  symex_run::equation two("int main(void) { int x = 2; return x + 1; }");
  REQUIRE(crcs(one.get()) != crcs(two.get()));
  REQUIRE(canonical(one.get()) != canonical(two.get()));
}

TEST_CASE("two runs of a branching program agree", "[symex][determinism]")
{
  require_identical(R"(
int nondet_int(void);
int main(void)
{
  int x = nondet_int();
  int y = nondet_int();
  if (nondet_int() > 0)
    x = nondet_int();
  else
    y = nondet_int();
  if (x > y)
    x = y;
  return x + y;
}
)");
}

TEST_CASE("two runs of a loop with calls agree", "[symex][determinism]")
{
  require_identical(R"(
int nondet_int(void);
int step(int v) { return v > 0 ? v - 1 : v + 1; }
int main(void)
{
  int total = 0;
  for (int i = 0; i < 3; i++)
  {
    int v = nondet_int();
    if (v > 0)
      total += step(v);
    else
      total -= step(v);
  }
  return total;
}
)");
}

TEST_CASE(
  "two runs over addressed locals agree once object ids are normalised",
  "[symex][determinism]")
{
  // A pointer whose target is branch-dependent: the value-set merged at the
  // join is keyed on expressions, so a set iterated in address order would
  // reorder the constraints recorded here.
  require_identical_modulo_object_ids(R"(
int nondet_int(void);
struct pair
{
  int a;
  int b;
};
int main(void)
{
  struct pair p = {nondet_int(), nondet_int()};
  int arr[4] = {nondet_int(), nondet_int(), nondet_int(), nondet_int()};
  int *q = nondet_int() > 0 ? &p.a : &p.b;
  int *r = nondet_int() > 0 ? &arr[1] : q;
  *r = nondet_int();
  return p.a + p.b + arr[*q & 3] + *r;
}
)");
}

TEST_CASE(
  "two runs of a heap program agree once object ids are normalised",
  "[symex][determinism]")
{
  // Dynamic allocation writes the validity arrays `dynamic_allocation.cpp`
  // maintains; those are indexed by allocation order, which must not drift.
  require_identical_modulo_object_ids(R"(
int nondet_int(void);
void *malloc(unsigned long);
void free(void *);
int main(void)
{
  int *p = (int *)malloc(sizeof(int) * 4);
  int *q = (int *)malloc(sizeof(int));
  if (!p || !q)
    return 0;
  p[nondet_int() & 3] = nondet_int();
  *q = p[0];
  free(p);
  free(q);
  return 0;
}
)");
}

TEST_CASE(
  "two runs minting an invalid_object agree byte for byte (R15)",
  "[symex][determinism]")
{
  // The heap case below pins `dynamic_counter` only; `invalid_counter` is a
  // separate counter with a separate reset, and nothing here mints a
  // `symex::invalid_object` unless a dereference cannot be resolved. An
  // external function's result is such a pointer.
  const std::string src = R"(
int *ext(void);
int nondet_int(void);
int main(void)
{
  int *p = ext();
  *p = nondet_int();
  return *p;
}
)";

  const std::vector<step_crc> first = crcs(symex_run::equation(src).get());
  const std::vector<step_crc> second = crcs(symex_run::equation(src).get());

  REQUIRE(first.size() == second.size());
  REQUIRE(first == second);
}

TEST_CASE(
  "two runs of a heap program agree byte for byte (R15)",
  "[symex][determinism]")
{
  // The strict form of the two cases above, and the one that would fail if
  // R15 came back: `dynamic_counter` and `invalid_counter` are reset per
  // exploration by `setup_for_new_explore`, so a second run in this process
  // numbers its objects from zero again and no canonicalisation is needed.
  // This case previously asserted the opposite -- it pinned the leak -- so a
  // regression flips it rather than merely weakening it.
  const std::string src = R"(
int nondet_int(void);
void *malloc(unsigned long);
int main(void)
{
  int *p = (int *)malloc(sizeof(int) * 4);
  if (!p)
    return 0;
  p[0] = nondet_int();
  return p[0];
}
)";

  const std::vector<step_crc> first = crcs(symex_run::equation(src).get());
  const std::vector<step_crc> second = crcs(symex_run::equation(src).get());

  REQUIRE(first.size() == second.size());
  REQUIRE(first == second);
}
