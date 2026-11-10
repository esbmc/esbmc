// A std::bad_alloc out of Bitwuzla's check_sat can leave its bit-blaster's
// unique table inconsistent, and destroying the solver afterwards crashed in
// AigNodeUniqueTable::erase (#10054). Each child process below fails one
// large allocation of the solve, catches the exception as bmct::run does and
// then destroys the solver; none may die of SIGSEGV. A child may still abort:
// a bad_alloc thrown through a noexcept frame inside the solver calls
// std::terminate before any caller can see it.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <memory>
#include <new>
#include <sys/wait.h>
#include <unistd.h>
#include <irep2/irep2_utils.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/solve.h>
#include <util/arith/arith_tools.h>
#include <util/config/config.h>
#include <util/config/options.h>
#include <util/lang/c_types.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>

// Only allocations of at least this size count: the vulnerable states come
// from the bit-blaster's table and node-vector reallocations, and skipping the
// many small node allocations keeps the sweep short.
static constexpr std::size_t large = 4096;
static long fail_at = -1;
static long large_allocs = 0;

void *operator new(std::size_t n)
{
  if (fail_at >= 0 && n >= large && large_allocs++ == fail_at)
    throw std::bad_alloc();
  if (void *p = std::malloc(n ? n : 1))
    return p;
  throw std::bad_alloc();
}

// noinline: GCC otherwise inlines these into library code and takes the
// free() of memory from the replaced operator new for a mismatch.
[[gnu::noinline]] void operator delete(void *p) noexcept
{
  std::free(p);
}

[[gnu::noinline]] void operator delete(void *p, std::size_t) noexcept
{
  std::free(p);
}

static bool solve_failing_at(long k)
{
  contextt ctx;
  namespacet ns(ctx);
  optionst options;
  std::unique_ptr<smt_convt> solver{create_solver("bitwuzla", ns, options)};

  const type2tc u32 = get_uint32_type();
  expr2tc x = symbol2tc(u32, "x");
  for (unsigned i = 0; i < 8; ++i)
  {
    expr2tc d = symbol2tc(u32, "d" + std::to_string(i));
    solver->assert_expr(notequal2tc(d, gen_zero(u32)));
    x = div2tc(u32, mul2tc(u32, x, d), d);
  }
  solver->assert_expr(equality2tc(x, from_integer(12345, u32)));

  large_allocs = 0;
  fail_at = k;
  bool threw = false;
  try
  {
    solver->dec_solve();
  }
  catch (const std::bad_alloc &)
  {
    threw = true;
  }
  fail_at = -1;
  return threw;
}

SCENARIO(
  "a Bitwuzla solve that runs out of memory can be destroyed",
  "[solvers][bitwuzla]")
{
  config.ansi_c.set_data_model(configt::LP64);

  REQUIRE_FALSE(solve_failing_at(LONG_MAX));
  const long points = large_allocs;
  REQUIRE(points > 0);

  long caught = 0, crashed = 0, first_caught = -1;
  for (long k = 0; k < points; ++k)
  {
    // An aborting child would flush the output this process buffered.
    std::fflush(nullptr);
    const pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0)
    {
      // Report a child's crash to the parent as its signal, not through the
      // fatal-condition handler and alternate stack Catch2 installed.
      for (int sig : {SIGABRT, SIGSEGV, SIGBUS})
        signal(sig, SIG_DFL);
      stack_t off = {};
      off.ss_flags = SS_DISABLE;
      sigaltstack(&off, nullptr);
      dup2(open("/dev/null", O_WRONLY), STDERR_FILENO);
      _exit(solve_failing_at(k) ? 0 : 1);
    }

    int status;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
    {
      if (caught++ == 0)
        first_caught = k;
    }
    else if (
      WIFSIGNALED(status) &&
      (WTERMSIG(status) == SIGSEGV || WTERMSIG(status) == SIGBUS))
      ++crashed;
  }

  INFO(caught << " of " << points << " failure points caught");
  REQUIRE(caught > 0);
  REQUIRE(crashed == 0);

  // A child writes no coverage profile, so repeat a point it survived here.
  REQUIRE(solve_failing_at(first_caught));
}
