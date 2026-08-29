/*******************************************************************
 Module: The MPOR dependency chain on the real engine

 Tier B of docs/roadmap/goto-symex-verification-plan.md (H-A6's A6.4, M6).

 `calculate_mpor_constraints` implements the recurrence of Kahlon, Wang and
 Gupta's monotonic partial order reduction. Of its five rules, the one that
 *removes* relations is the reset of the active thread's row, and A6.4 asks
 whether that reset can lose a chain the reduction later needs -- a lost chain
 blocks a transition, which drops interleavings with no diagnostic.

 The reset is faithful: DCil records a chain from the last transition of Tl to
 Ti's, and the transition just taken is the latest in the run, so no chain can
 leave it. That argument is now the engine's own invariant -- every 1 in the
 chain points forward in the run order the engine records alongside it -- and
 this file re-checks it from outside, on the chain a real exploration produces.
 What is *not* a transcription of the recurrence is the guard on the un-run
 case, which tests the column entry DCja where MPOR tests the diagonal DCjj;
 that equivalence is a property of the run and is asserted in the engine.
 This file pins the chain's shape after every interleaving of a real concurrent
 program. It does *not* discriminate the un-run guard itself: reverting that
 guard leaves every assertion here green, and the interleaving count unchanged
 at 49, so the shape is what it pins and nothing more.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <algorithm>
#include <string>
#include <vector>

#include <goto-symex/reachability_tree.h>
#include <goto-symex/symex_target_equation.h>
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
    REQUIRE(prog.functions.function_map.size() > 0);
    rt.setup_for_new_explore();
  }

  struct chaint
  {
    const std::vector<std::vector<int>> &dc;
    const std::vector<unsigned int> &order;
  };

  chaint run_one_interleaving()
  {
    rt.get_next_formula();
    return {
      rt.get_cur_state().get_dependency_chain(),
      rt.get_cur_state().get_thread_last_transition()};
  }

  bool more_interleavings()
  {
    return rt.setup_next_formula();
  }

private:
  std::string source;
  program prog;
  namespacet ns;
  optionst opts;
  reachability_treet rt;
};

/// Chain shapes observed across an exploration, so a check cannot pass by
/// never meeting the case it describes.
struct census
{
  unsigned interleavings = 0;
  unsigned run_threads = 0;
  unsigned chains = 0;
  unsigned no_relation = 0;
  unsigned rows_reset_after_running = 0;
};

void check_chain(const engine::chaint &c, census &seen)
{
  const std::vector<std::vector<int>> &dc = c.dc;
  const size_t n = dc.size();
  REQUIRE(n >= 2);
  REQUIRE(c.order.size() == n);

  // A6.4 from outside the engine: a chain into Tj's last transition can only
  // start at one that preceded it. The reset of the active row is what keeps
  // this true, so a run where some thread took a second transition after a
  // chain out of it was recorded is a run where the reset had work to do --
  // counted below, since without it the property is unobservable.
  unsigned latest = 0;
  for (size_t j = 0; j < n; j++)
    latest = std::max(latest, c.order[j]);
  if (latest > n)
    seen.rows_reset_after_running++;

  for (size_t j = 0; j < n; j++)
  {
    REQUIRE(dc[j].size() == n);
    REQUIRE((dc[j][j] == 1) == (c.order[j] != 0));

    bool row_all_zero = true;
    for (size_t l = 0; l < n; l++)
    {
      REQUIRE(dc[j][l] >= -1);
      REQUIRE(dc[j][l] <= 1);
      row_all_zero &= dc[j][l] == 0;

      if (j == l)
        continue;
      if (dc[j][l] == 1)
      {
        REQUIRE(c.order[j] < c.order[l]);
        seen.chains++;
      }
      else if (dc[j][l] == -1)
        seen.no_relation++;
    }

    // The diagonal is the run marker: a thread that has taken a transition
    // depends on itself, and one that has not has no relation to anything.
    // The converse of the second half does not hold -- a run thread's row
    // carries 0 in the column of any thread created after it last ran, which
    // is why the un-run test upstream has to be the diagonal.
    REQUIRE((dc[j][j] == 0 || dc[j][j] == 1));
    if (dc[j][j] == 0)
      REQUIRE(row_all_zero);
    else
      seen.run_threads++;
  }
}

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
} // namespace

TEST_CASE(
  "the MPOR dependency chain keeps its shape across an exploration",
  "[goto-symex][mpor]")
{
  engine e(TWO_WRITERS);
  census seen;

  do
  {
    check_chain(e.run_one_interleaving(), seen);
    seen.interleavings++;
  } while (seen.interleavings < 64 && e.more_interleavings());

  // Anti-vacuity: the exploration has to have reached the shapes the checks
  // above discriminate between, or they hold for want of a subject.
  REQUIRE(seen.interleavings >= 2);
  REQUIRE(seen.run_threads > 0);
  REQUIRE(seen.chains > 0);
  REQUIRE(seen.no_relation > 0);
  REQUIRE(seen.rows_reset_after_running > 0);
}
