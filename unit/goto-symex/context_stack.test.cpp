/*******************************************************************
 Module: The incremental context stack's balance on the real engine

 H-B7's residual row of docs/roadmap/goto-symex-verification-plan.md §7.3:
 H-A8 assumes `push_ctx`/`pop_ctx` calls are balanced by the caller.

 Under `--smt-during-symex` the equation is a `runtime_encoded_equationt` shared
 by every execution state, and each state's scope is a solver context: a clone
 pushes, a destructor pops (`execution_state.cpp:1487-1501`). Nothing pairs them
 structurally -- the initial state is *constructed* rather than cloned, so it
 has no push of its own, and `reachability_treet` compensates with one explicit
 `targ->push_ctx()` at the end of `setup_for_new_explore` ("Start with a depth
 of 1"). The balance is therefore a property of two files agreeing, which is
 what makes it an assumption worth checking rather than reading. It holds: one
 push at setup plus one per clone, one pop per destruction, so an exhausted
 exploration lands back on zero -- the setup push is consumed by the initial
 state's own destructor, which is the pop that would otherwise have no partner.

 It is worth checking because an unbalanced pop is undefined behaviour, not a
 diagnostic: `pop_ctx` takes `scoped_end_points.back()` on a list the
 constructor leaves empty (`symex_target_equation.cpp:527-537, 596-609`). A
 missing push does not produce a wrong answer here, it corrupts the solver
 stack, so this file also fails by crashing -- which is a signal, not a flaw in
 it.

 Per §6.1 the equation and the solver are the real ones. A counting subclass
 standing in for `runtime_encoded_equationt` would be a test double, and would
 also not exercise the code whose stack is at issue.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include <goto-symex/reachability_tree.h>
#include <goto-symex/symex_target_equation.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/solve.h>
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
        goto_factory::get_default_cmdline("test.c")))
  {
    // Read by execution_statet's constructor, so it has to be set before the
    // reachability tree builds the initial state.
    opts.set_option("smt-during-symex", true);

    solver.reset(create_solver("", ns, opts));
    REQUIRE(solver != nullptr);
    target = std::make_shared<runtime_encoded_equationt>(ns, *solver);

    rt = std::make_unique<reachability_treet>(
      prog.functions, ns, opts, target, prog.context);
  }

  void setup()
  {
    rt->setup_for_new_explore();
  }

  void run_one_interleaving()
  {
    live = std::dynamic_pointer_cast<runtime_encoded_equationt>(
      rt->get_next_formula().target);
    REQUIRE(live != nullptr);
  }

  bool more_interleavings()
  {
    return rt->setup_next_formula();
  }

  /// Scopes open on the equation the exploration actually drives. Not the
  /// template: setup_for_new_explore clones that, and pushes on the clone.
  size_t depth() const
  {
    REQUIRE(live != nullptr);
    return live->scoped_end_points.size();
  }

  /// Scopes open on the template this test constructed.
  size_t template_depth() const
  {
    return target->scoped_end_points.size();
  }

private:
  std::string source;
  program prog;
  namespacet ns;
  optionst opts;
  std::unique_ptr<smt_convt> solver;
  std::shared_ptr<runtime_encoded_equationt> target;
  std::unique_ptr<reachability_treet> rt;
  std::shared_ptr<runtime_encoded_equationt> live;
};

// Two threads racing on two globals: enough interleavings that the exploration
// clones states repeatedly, which is what opens scopes in the first place.
const char *TWO_WRITERS = R"(
#include <pthread.h>

int g;
int h;

void *writer(void *arg)
{
  g = 1;
  h = g;
  return 0;
}

void *reader(void *arg)
{
  h = 2;
  g = h;
  return 0;
}

int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, writer, 0);
  pthread_create(&b, 0, reader, 0);
  return 0;
}
)";

const char *STRAIGHT_LINE = R"(
int nondet_int(void);
int main(void)
{
  int x = nondet_int();
  int y = x + 1;
  __ESBMC_assert(y != x, "no overflow here");
  return 0;
}
)";
} // namespace

TEST_CASE(
  "the explored equation is a clone, not the template",
  "[symex][context-stack]")
{
  engine e(STRAIGHT_LINE);

  // The constructor leaves scoped_end_points empty -- the state whose pop_ctx
  // reads out of bounds -- and setup_for_new_explore does not change that for
  // the template, because it clones it and pushes on the clone
  // (reachability_tree.cpp:330-339, symex_target_equation.cpp:633-645). A
  // balance check that watched the object the caller constructed would see
  // nothing at all, so this is stated before relying on it.
  REQUIRE(e.template_depth() == 0);
  e.setup();
  REQUIRE(e.template_depth() == 0);

  e.run_one_interleaving();
  REQUIRE(e.depth() == 1);
  REQUIRE(e.template_depth() == 0);
}

TEST_CASE(
  "the setup push is what the initial state's pop consumes",
  "[symex][context-stack]")
{
  engine e(STRAIGHT_LINE);
  e.setup();
  e.run_one_interleaving();

  // reachability_tree.cpp's "Start with a depth of 1": the push standing in for
  // the initial state, which was constructed rather than cloned. It is open for
  // as long as that state is.
  REQUIRE(e.depth() == 1);

  // Nothing else to explore; retiring the initial state pops it, and the pair
  // is what keeps the count off zero while a state is alive.
  REQUIRE_FALSE(e.more_interleavings());
  REQUIRE(e.depth() == 0);
}

TEST_CASE(
  "a concurrent exploration returns to the setup depth",
  "[symex][context-stack]")
{
  engine e(TWO_WRITERS);
  e.setup();

  size_t interleavings = 0;
  size_t max_depth = 1;
  bool exhausted = false;
  while (interleavings < 256)
  {
    e.run_one_interleaving();
    max_depth = std::max(max_depth, e.depth());
    interleavings++;
    if (!e.more_interleavings())
    {
      exhausted = true;
      break;
    }
  }

  // The depth mid-exploration is the DFS stack, so the balance below only means
  // anything once every state has been retired. Stopping at a cap instead would
  // read the open scopes of live states as an imbalance.
  INFO("interleavings " << interleavings << ", max depth " << max_depth);
  REQUIRE(exhausted);

  // Anti-vacuity: balance is trivial if nothing was ever pushed. The
  // exploration has to have cloned states and opened scopes for the depth
  // below to be saying anything.
  REQUIRE(interleavings > 1);
  REQUIRE(max_depth > 1);

  // Balanced: one push at setup plus one per clone, one pop per destruction,
  // and the initial state is destroyed like any other, so an exhausted
  // exploration lands on zero rather than on the setup push.
  REQUIRE(e.depth() == 0);
}
