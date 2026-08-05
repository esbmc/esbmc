/*******************************************************************
 Module: Value-set merge monotonicity on the real engine

 Tier B of docs/roadmap/goto-symex-verification-plan.md (H-B6, I9).

 §4.3's I9: merging two paths must *union* their points-to information. An
 accidental intersection is silent -- the value set only feeds dereference
 resolution, so a dropped target does not fail anywhere, it just stops being
 considered, and every check on that object disappears with it. That is the
 missed-bug direction with no diagnostic, which is why H-B6 asserts the
 post-merge set contains both inputs rather than trusting `make_union`'s name.

 **Mutation testing, and the two mutants it separates.** Deleting the
 `make_union` call from `merge_value_sets` leaves every case green, including
 `early_exit`, whose arms reach the join by different routes. Instrumenting the
 call explains it: over that program the union arm runs three times and returns
 `changed == false` every time, and the `guard.is_false()` replacement arm above
 it is never taken at all. Guarded assignment *adds* to a pointer's object map
 rather than replacing it, and `cur_state`'s value set is never rewound when a
 branch is abandoned, so both targets are already present before any join runs.
 Deleting the union is therefore an over-approximation of an
 over-approximation -- it cannot lose a target, which is why no case detects it.

 Replacing the union with an *intersection* is the mutant that matters, and it
 is caught: three of the five cases fail. That is I9's actual content -- a merge
 must not shrink the points-to set -- so these cases do discharge it, and the
 surviving deletion mutant is not evidence against them.

 The consequence for readers of `merge_value_sets`: its union is redundant at
 every join reachable here, so it is load-bearing only against a future change
 that makes value sets path-sensitive. Note also that `make_union`'s `keepnew`
 parameter selects whether an entry present only in the source survives;
 symex passes `true`, while the static `value_set_domaint::merge` does not.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <set>
#include <string>
#include <utility>

#include <goto-symex/reachability_tree.h>
#include <pointer-analysis/value_set.h>
#include <util/symtab/namespace.h>

#include "../testing-utils/goto_factory.h"

namespace
{
class engine
{
public:
  explicit engine(std::string src)
    : source(std::move(src)),
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
  }

  void run()
  {
    REQUIRE(rt.get_next_formula().target != nullptr);
  }

  const value_sett &value_set()
  {
    return rt.get_cur_state().get_active_state().value_set;
  }

private:
  std::string source;
  program prog;
  namespacet ns;
  optionst opts;
  reachability_treet rt;
};

/** Names of the objects the entry named exactly `id` may point to. The entry
 *  match is exact and required to be unique: a substring match on "p" also
 *  catches main's `$tmp::return_value$` temporaries, and a max over those
 *  passes this test without ever inspecting the pointer.
 *
 *  The *targets* are returned by name rather than counted. A global pointer's
 *  object map already holds its zero-initialiser before either arm runs, so a
 *  cardinality of two is reached without any merge happening at all -- which is
 *  what let the first version of these cases survive deleting `make_union`. */
std::set<std::string> targets_of(const value_sett &vs, const std::string &id)
{
  size_t matches = 0;
  std::set<std::string> targets;
  for (const auto &entry : vs.values)
    if (entry.second.identifier == id)
    {
      ++matches;
      for (const auto &object : entry.second.object_map)
      {
        const expr2tc &o = value_sett::object_numbering[object.first];
        targets.insert(
          is_symbol2t(o) ? to_symbol2t(o).thename.as_string() : get_expr_id(o));
      }
    }
  REQUIRE(matches == 1);
  return targets;
}

bool points_to(const std::set<std::string> &targets, const std::string &object)
{
  for (const std::string &t : targets)
    if (t.find(object) != std::string::npos)
      return true;
  return false;
}

// Two arms give one global pointer two different targets. `p` is a global so
// its entry outlives main's frame, which is popped by the time run() returns.
const char *two_arms = R"(
int a, b;
int *p;
int nondet_int(void);
int main(void)
{
  if (nondet_int())
    p = &a;
  else
    p = &b;
  return *p;
}
)";

// Control: one arm leaves `p` untouched, so the merge must union the arm that
// wrote it with the fall-through that did not. Losing either direction here is
// the same defect as losing an arm above.
const char *one_arm = R"(
int a;
int *p;
int nondet_int(void);
int main(void)
{
  p = 0;
  if (nondet_int())
    p = &a;
  return p == 0 ? 0 : *p;
}
)";

// A join whose arms arrive by different routes: one leaves the `if` by its own
// jump, the other falls through. Included because it is the shape that *would*
// distinguish a redundant union from a load-bearing one if value sets were
// rewound on an abandoned branch -- they are not, see the header.
const char *early_exit = R"(
int a, b;
int *p;
int nondet_int(void);
int main(void)
{
  if (nondet_int())
  {
    p = &a;
    goto join;
  }
  p = &b;
join:
  return *p;
}
)";
} // namespace

TEST_CASE("a two-arm join keeps both arms' targets", "[symex][value-set]")
{
  engine e(two_arms);
  e.run();

  // Losing a name here is an arm dropped at the join: an intersection, not a
  // union. Both &a and &b are reachable, so both must survive.
  const std::set<std::string> targets = targets_of(e.value_set(), "c:@p");
  REQUIRE(points_to(targets, "@a"));
  REQUIRE(points_to(targets, "@b"));
}

TEST_CASE("a one-armed join keeps the fall-through value", "[symex][value-set]")
{
  engine e(one_arm);
  e.run();

  const std::set<std::string> targets = targets_of(e.value_set(), "c:@p");
  REQUIRE(points_to(targets, "@a"));
}

TEST_CASE("a join whose arms diverge keeps both targets", "[symex][value-set]")
{
  engine e(early_exit);
  e.run();

  const std::set<std::string> targets = targets_of(e.value_set(), "c:@p");
  REQUIRE(points_to(targets, "@a"));
  REQUIRE(points_to(targets, "@b"));
}

TEST_CASE("make_union never shrinks the destination", "[symex][value-set]")
{
  engine e(two_arms);
  e.run();

  const value_sett &produced = e.value_set();

  // Unioning with a copy of itself is the identity on a monotone merge; on an
  // intersecting one it is the first thing to lose entries.
  value_sett self = produced;
  self.make_union(produced, true);

  REQUIRE(self.values.size() >= produced.values.size());
  for (const auto &entry : produced.values)
  {
    auto it = self.values.find(entry.first);
    REQUIRE(it != self.values.end());
    REQUIRE(it->second.object_map.size() >= entry.second.object_map.size());
  }
}

TEST_CASE("make_union with an empty set loses nothing", "[symex][value-set]")
{
  engine e(two_arms);
  e.run();

  value_sett merged = e.value_set();
  const size_t before = merged.values.size();
  REQUIRE(before > 0);

  value_sett empty = e.value_set();
  empty.values.clear();
  merged.make_union(empty, true);

  REQUIRE(merged.values.size() == before);
}
