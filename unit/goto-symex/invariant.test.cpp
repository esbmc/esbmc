/*******************************************************************
 Module: Release-mode enforcement of goto-symex's state invariants

 Tier B of docs/roadmap/goto-symex-verification-plan.md (R1, M3). Each case
 violates one invariant on the real engine state and requires the process to die
 with our own diagnostic: matching the message is what distinguishes a live
 SYMEX_INVARIANT from a libc `assert` that NDEBUG has already removed.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <array>
#include <functional>
#include <string>

// Windows lacks fork() and <sys/wait.h>, so the death tests are POSIX-only.
#if !defined(_WIN32)
#  include <csignal>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

#include <goto-symex/reachability_tree.h>
#include <goto-symex/renaming.h>
#include <irep2/irep2_expr.h>
#include <util/lang/c_types.h>
#include <util/symtab/namespace.h>

#include "../testing-utils/goto_factory.h"

namespace
{
class engine
{
public:
  engine()
    : prog(goto_factory::get_goto_functions(
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
    rt.setup_for_new_explore();
  }

  goto_symex_statet &state()
  {
    return rt.get_cur_state().get_active_state();
  }

  /** Symex to completion; leaves __ESBMC_main + main on the call stack. */
  void run()
  {
    REQUIRE(rt.get_next_formula().target != nullptr);
  }

private:
  std::string source = "int main(void) { int x = 0; return x; }";
  program prog;
  namespacet ns;
  optionst opts;
  reachability_treet rt;
};

expr2tc l1_symbol(
  const char *name,
  symbol_renaming_level lev = symbol_renaming_level::level1)
{
  return symbol2tc(int_type2(), irep_idt(name), lev, 0, 0, 0, 0);
}

#if !defined(_WIN32)
/** Require `fn` to abort, and its diagnostic to name the invariant. */
void require_invariant_abort(
  const std::function<void()> &fn,
  const std::string &reason)
{
  int pipe_fds[2];
  REQUIRE(pipe(pipe_fds) == 0);

  pid_t pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0)
  {
    close(pipe_fds[0]);
    if (dup2(pipe_fds[1], STDERR_FILENO) < 0)
      _exit(2);
    if (!freopen("/dev/null", "w", stdout))
      _exit(2);
    fn();
    _exit(0); // unreachable if fn aborts
  }

  close(pipe_fds[1]);
  std::string captured;
  std::array<char, 512> buffer;
  // Bounded by the destination's own size, and its result bounds the append.
  // flawfinder: ignore
  for (ssize_t n; (n = read(pipe_fds[0], buffer.data(), buffer.size())) > 0;)
    captured.append(buffer.data(), n);
  close(pipe_fds[0]);

  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  REQUIRE(WIFSIGNALED(status));
  REQUIRE(WTERMSIG(status) == SIGABRT);
  // A libc assert under -UNDEBUG would also abort, but with its own wording.
  REQUIRE(captured.find("goto-symex invariant violated") != std::string::npos);
  REQUIRE(captured.find(reason) != std::string::npos);
}
#endif
} // namespace

TEST_CASE("a well-formed run trips no invariant", "[symex][invariant]")
{
  // Non-vacuity: the promoted checks are on paths symex takes constantly.
  engine e;
  e.run();
  goto_symex_statet &state = e.state();

  REQUIRE(state.call_stack.size() >= 2);
  REQUIRE(
    &state.previous_frame() == &state.call_stack[state.call_stack.size() - 2]);

  const size_t before = state.call_stack.size();
  state.pop_frame();
  REQUIRE(state.call_stack.size() == before - 1);

  expr2tc lhs = l1_symbol("invariant_test_var");
  state.level2.make_assignment(lhs, expr2tc(), expr2tc());
  REQUIRE(to_symbol2t(lhs).level2_num == 1);
}

#if !defined(_WIN32)
TEST_CASE("previous_frame with no caller aborts (R7)", "[symex][invariant]")
{
  // Pre-M3 this formed `begin() - 1` and read it: undefined by [expr.add]/4.
  require_invariant_abort(
    []() {
      engine e;
      goto_symex_statet &state = e.state();
      while (state.call_stack.size() > 1)
        state.pop_frame();
      state.previous_frame();
    },
    "no caller frame beneath the current one");
}

TEST_CASE(
  "popping a frame with pending merges aborts (I6/R2)",
  "[symex][invariant]")
{
  // The snapshots are paths a join still owes; dropping them was silent (R2).
  require_invariant_abort(
    []() {
      engine e;
      goto_symex_statet &state = e.state();
      state.top().merge_state_map[state.source.pc];
      state.pop_frame();
    },
    "unmerged path snapshots");
}

TEST_CASE("an L2 counter moving backwards aborts (I1)", "[symex][invariant]")
{
  // Reissuing an index lets two program values share one SSA name.
  require_invariant_abort(
    []() {
      engine e;
      expr2tc lhs = l1_symbol("invariant_test_var");
      e.state().level2.make_assignment(lhs, expr2tc(), expr2tc());

      expr2tc l1 = l1_symbol("invariant_test_var");
      e.state().level2.rename(l1, 0);
    },
    "L2 assignment counter moved backwards");
}

TEST_CASE("assigning to a non-L1 name aborts (I2)", "[symex][invariant]")
{
  // An L0 lhs keys a different entry than the one the caller publishes from.
  require_invariant_abort(
    []() {
      engine e;
      expr2tc lhs =
        l1_symbol("invariant_test_var", symbol_renaming_level::level0);
      e.state().level2.make_assignment(lhs, expr2tc(), expr2tc());
    },
    "L2 assignment counters are keyed by the L1 name");
}
#endif
