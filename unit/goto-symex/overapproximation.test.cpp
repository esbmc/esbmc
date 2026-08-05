/*******************************************************************
 Module: R9's "sound over-approximation" claims, as checkable predicates

 Tier B of docs/roadmap/goto-symex-verification-plan.md (§9.2 R9).

 R9 records three places where a comment argues an approximation is sound and
 nothing checks it. A comment is not a proof, but it is also not the thing to
 verify: what a harness can do is state each claim as a predicate over the
 produced equation and pin the direction, so a change that quietly reverses it
 fails here rather than in a verdict months later.

 Two of the three are pinned below. The third -- the value-set filter after a
 pointer havoc (`symex_assign.cpp:554-576`) -- is not, and the reason is
 recorded in §15 rather than hidden: reaching it needs
 `pc->inductive_step_instruction`, which only the k-induction goto transform
 sets, so it is out of reach of a `goto_factory` program.

 The direction that matters for both claims below is *never adding behaviour*.
 Dropping a constraint is safe; dropping a call target, or letting a discarded
 body run, is not.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <string>

#include "symex_run.h"

namespace
{
/// Does any expression in this step contain an uninterpreted-function term?
bool mentions_uninterpreted_func(const expr2tc &e)
{
  if (is_nil_expr(e))
    return false;
  if (is_uninterpreted_func2t(e))
    return true;

  bool found = false;
  e->foreach_operand([&found](const expr2tc &op) {
    if (!found)
      found = mentions_uninterpreted_func(op);
  });
  return found;
}

size_t count_uninterpreted_funcs(const symex_target_equationt &eq)
{
  size_t n = 0;
  for (const auto &step : eq.SSA_steps)
    if (mentions_uninterpreted_func(step.rhs))
      n++;
  return n;
}

/// Steps assigning a name belonging to `function` -- its parameters and locals.
/// Non-zero means the call was dispatched and its body symexed.
/// How many steps assign a name containing `name`. Globals carry exactly one,
/// their zero-initialiser, unless something writes them.
size_t assignments_to(const symex_target_equationt &eq, const std::string &name)
{
  size_t n = 0;
  for (const auto &step : eq.SSA_steps)
  {
    if (
      !step.is_assignment() || is_nil_expr(step.lhs) || !is_symbol2t(step.lhs))
      continue;
    if (to_symbol2t(step.lhs).get_symbol_name().find(name) != std::string::npos)
      n++;
  }
  return n;
}

size_t
steps_inside(const symex_target_equationt &eq, const std::string &function)
{
  const std::string tag = "@F@" + function + "@";
  size_t n = 0;
  for (const auto &step : eq.SSA_steps)
  {
    if (
      !step.is_assignment() || is_nil_expr(step.lhs) || !is_symbol2t(step.lhs))
      continue;
    if (to_symbol2t(step.lhs).get_symbol_name().find(tag) != std::string::npos)
      n++;
  }
  return n;
}

// A scalar signature: the case the fallback is *not* for. Both calls take the
// same argument, so a congruence constraint is available to be emitted.
const char *scalar_uf = R"(
int __ESBMC_uninterpreted_scalar(int x);
int main(void)
{
  int a = __ESBMC_uninterpreted_scalar(3);
  int b = __ESBMC_uninterpreted_scalar(3);
  return a + b;
}
)";

// A pointer argument makes the signature non-scalar, so the result falls back
// to a fresh nondet. The body writes a global: "the body is still discarded"
// is the half of the claim that would be unsound if it were wrong, and `side`
// is how this file observes it.
const char *nonscalar_uf = R"(
int side;
int __ESBMC_uninterpreted_ptr(int *p)
{
  side = 1;
  return *p;
}
int g;
int main(void)
{
  int a = __ESBMC_uninterpreted_ptr(&g);
  int b = __ESBMC_uninterpreted_ptr(&g);
  return a + b;
}
)";

// `two` is address-taken with a cast, so the value set of `p` lists a target of
// the wrong arity. Dispatching it would nondet-fill the missing argument.
const char *mixed_arity_targets = R"(
int nondet_int(void);
int one(int a)
{
  return a;
}
int two(int a, int b)
{
  return a + b;
}
typedef int (*fp1)(int);
int main(void)
{
  fp1 p;
  if (nondet_int())
    p = one;
  else
    p = (fp1)two;
  return p(5);
}
)";

// Every candidate is incompatible with the declared type. The filter would
// empty the list, so it must keep it: dropping every target skips the call and
// silently havocs its result, which is the missed-bug direction.
const char *all_incompatible_targets = R"(
int nondet_int(void);
int two(int a, int b)
{
  return a + b;
}
int three(int a, int b, int c)
{
  return a + b + c;
}
typedef int (*fp1)(int);
int main(void)
{
  fp1 p;
  if (nondet_int())
    p = (fp1)two;
  else
    p = (fp1)three;
  return p(5);
}
)";
} // namespace

TEST_CASE(
  "a scalar uninterpreted function is encoded as one",
  "[symex][overapproximation]")
{
  symex_run::equation run(scalar_uf);

  // The control for the fallback below: without this, "no uninterpreted_func2t
  // was emitted" would hold for a program where none was ever due.
  REQUIRE(count_uninterpreted_funcs(run.get()) >= 2);
}

TEST_CASE(
  "a non-scalar signature falls back without running the body",
  "[symex][overapproximation]")
{
  symex_run::equation run(nonscalar_uf);
  const symex_target_equationt &eq = run.get();

  // The fallback fired: no uninterpreted-function term survives.
  REQUIRE(count_uninterpreted_funcs(eq) == 0);

  // ... and it dropped only the congruence constraint. The body is still
  // discarded, so its write to `side` never happens. If this fails the
  // approximation is not weaker than the uninterpreted-function semantics, it
  // is different from them, and R9's "never adding behaviour" is false.
  // `side` keeps exactly the one assignment every global gets, its
  // zero-initialiser. A second one would be the discarded body running.
  REQUIRE(assignments_to(eq, "c:@side") == 1);
  REQUIRE(steps_inside(eq, "__ESBMC_uninterpreted_ptr") == 0);
}

TEST_CASE(
  "an incompatible call target is dropped, a compatible one is kept",
  "[symex][overapproximation]")
{
  symex_run::equation run(mixed_arity_targets);
  const symex_target_equationt &eq = run.get();

  REQUIRE(steps_inside(eq, "one") > 0);
  REQUIRE(steps_inside(eq, "two") == 0);
}

TEST_CASE(
  "a filter that would empty the target list keeps it",
  "[symex][overapproximation]")
{
  symex_run::equation run(all_incompatible_targets);
  const symex_target_equationt &eq = run.get();

  // No candidate matches the declared type, so the filter must not run: a call
  // dispatched to a wrong-arity target is a spurious counterexample, but a call
  // dispatched to nothing at all is a missed one.
  REQUIRE(steps_inside(eq, "two") + steps_inside(eq, "three") > 0);
}
