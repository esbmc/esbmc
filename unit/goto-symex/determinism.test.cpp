/*******************************************************************
 Module: Run-to-run determinism of the produced equation

 H-B2 of docs/roadmap/goto-symex-verification-plan.md (P10, M4).

 Objective 7: two symex runs over the same program in the same configuration
 must produce the same equation. This one is load-bearing for the rest of the
 plan rather than for any single verdict — a reordering invisible in any one run
 makes every other result here unreproducible and silently invalidates
 regression pinning.

 The property holds *modulo object numbering*, and not strictly: two counters
 that name symex objects are `static thread_local` and never reset, so a second
 run in the same process numbers its objects from where the first stopped
 (finding R15). The cases below therefore split — strict equality where no such
 object is created, equality after canonicalising the numbering where one is,
 and one case pinning the leak itself so it cannot widen unnoticed.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <goto-symex/reachability_tree.h>
#include <irep2/irep2_expr.h>
#include <util/symtab/namespace.h>

#include "../testing-utils/goto_factory.h"

namespace
{
/** One symex run, owning everything the equation outlives: `namespacet` is held
 *  by reference inside `symex_target_equationt`, so the bundle has to stay alive
 *  as long as the equation it produced. Comparing two runs means holding two of
 *  these at once, which is why this is not the `engine` the other Tier-B tests
 *  share — those expose the live state and own a single run. */
class run
{
public:
  explicit run(const std::string &src)
    : source(src),
      prog(goto_factory::get_goto_functions(
        source,
        goto_factory::Architecture::BIT_64)),
      ns(prog.context),
      opts(goto_factory::get_default_options(
        goto_factory::get_default_cmdline("test.c"))),
      rt(
        prog.functions,
        ns,
        opts,
        std::make_shared<symex_target_equationt>(ns),
        prog.context)
  {
    opts.set_option("unwind", "4");
    rt.setup_for_new_explore();
    equation = std::dynamic_pointer_cast<symex_target_equationt>(
      rt.get_next_formula().target);
    REQUIRE(equation != nullptr);
  }

  const symex_target_equationt &eq() const
  {
    return *equation;
  }

private:
  std::string source;
  program prog;
  namespacet ns;
  optionst opts;
  reachability_treet rt;
  std::shared_ptr<symex_target_equationt> equation;
};

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
  require_equal_steps(crcs(run(src).eq()), crcs(run(src).eq()));
}

void require_identical_modulo_object_ids(const std::string &src)
{
  require_equal_steps(canonical(run(src).eq()), canonical(run(src).eq()));
}
} // namespace

TEST_CASE("the comparators distinguish two programs", "[symex][determinism]")
{
  // The control for every case below: a comparator that cannot tell two
  // equations apart would pass all of them without checking anything.
  const std::string one = "int main(void) { int x = 1; return x; }";
  const std::string two = "int main(void) { int x = 2; return x + 1; }";
  REQUIRE(crcs(run(one).eq()) != crcs(run(two).eq()));
  REQUIRE(canonical(run(one).eq()) != canonical(run(two).eq()));
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

TEST_CASE("object numbering leaks across runs (R15)", "[symex][determinism]")
{
  // Pins the leak the two cases above normalise around:
  // execution_statet::dynamic_counter (execution_state.h) and
  // dereferencet::invalid_counter (dereference.h) are `static thread_local` and
  // reset nowhere — unlike the sibling nondet_count, a plain member the
  // constructor zeroes. The equation is therefore not a function of (program,
  // options) alone: it also depends on how many objects earlier runs in this
  // process created. Delete this case when the counters are reset per
  // exploration; the canonical-equality cases above are the durable property.
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

  const std::vector<step_crc> first = crcs(run(src).eq());
  const std::vector<step_crc> second = crcs(run(src).eq());
  REQUIRE(first.size() == second.size());
  REQUIRE(first != second);
}
