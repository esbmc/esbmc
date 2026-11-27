// A Bitwuzla solver freed neither its asts, its sorts nor the tuple flattener
// create_solver built for it, so every claim of a --multi-property run leaked
// them and a long run's footprint grew with each one. Counting the program's
// live allocations around whole solver lifetimes pins that a solver frees
// everything it made. Bitwuzla's own nodes come from malloc and are not
// counted; the asts holding them are.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <cstdlib>
#include <memory>
#include <new>
#include <irep2/irep2_utils.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/solve.h>
#include <util/arith/arith_tools.h>
#include <util/config/config.h>
#include <util/config/options.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>

static long live_allocations = 0;

void *operator new(std::size_t n)
{
  if (void *p = std::malloc(n ? n : 1))
  {
    ++live_allocations;
    return p;
  }
  throw std::bad_alloc();
}

// noinline: GCC otherwise inlines these into library code and takes the
// free() of memory from the replaced operator new for a mismatch.
[[gnu::noinline]] void operator delete(void *p) noexcept
{
  if (p)
    --live_allocations;
  std::free(p);
}

[[gnu::noinline]] void operator delete(void *p, std::size_t) noexcept
{
  if (p)
    --live_allocations;
  std::free(p);
}

static type2tc pair_type()
{
  const type2tc u32 = get_uint32_type();
  return struct_type2tc(
    std::vector<type2tc>{u32, u32},
    std::vector<irep_idt>{"a", "b"},
    std::vector<irep_idt>{"a", "b"},
    "pair",
    false);
}

// A struct operand puts the tuple flattener and its sorts to work too.
static void solve_once(const optionst &options)
{
  contextt ctx;
  namespacet ns(ctx);
  std::unique_ptr<smt_convt> solver{create_solver("bitwuzla", ns, options)};

  const type2tc u32 = get_uint32_type();
  const type2tc pair = pair_type();
  const expr2tc s = symbol2tc(pair, "s");
  const expr2tc x = symbol2tc(u32, "x");
  solver->assert_expr(
    equality2tc(member2tc(u32, s, "a"), add2tc(u32, x, from_integer(1, u32))));
  solver->assert_expr(notequal2tc(member2tc(u32, s, "b"), x));
  REQUIRE(solver->dec_solve() == P_SATISFIABLE);
}

// A zeroed two-dimensional struct array reaches the node flattener's
// tuple_array_of, which makes an array sort of its own.
static void solve_zeroed_table_once(const optionst &options)
{
  contextt ctx;
  namespacet ns(ctx);
  std::unique_ptr<smt_convt> solver{create_solver("bitwuzla", ns, options)};

  const type2tc u32 = get_uint32_type();
  const type2tc pair = pair_type();
  const type2tc row = array_type2tc(pair, from_integer(4, u32), false);
  const type2tc table = array_type2tc(row, from_integer(4, u32), false);
  const expr2tc zero = from_integer(0, u32);
  const expr2tc zeroed = constant_array_of2tc(
    table,
    constant_array_of2tc(
      row, constant_struct2tc(pair, std::vector<expr2tc>{zero, zero})));
  const expr2tc i = symbol2tc(u32, "i"), j = symbol2tc(u32, "j");
  const expr2tc a =
    member2tc(u32, index2tc(pair, index2tc(row, zeroed, i), j), "a");
  solver->assert_expr(lessthan2tc(i, from_integer(4, u32)));
  solver->assert_expr(lessthan2tc(j, from_integer(4, u32)));
  solver->assert_expr(notequal2tc(a, zero));
  REQUIRE(solver->dec_solve() == P_UNSATISFIABLE);
}

static void require_no_growth(
  const optionst &options,
  void (*solve)(const optionst &) = solve_once)
{
  config.ansi_c.set_data_model(configt::LP64);
  // The first lifetimes fill process-wide caches (interned names, types).
  solve(options);
  solve(options);
  const long before = live_allocations;
  solve(options);
  solve(options);
  REQUIRE(live_allocations == before);
}

TEST_CASE("a Bitwuzla solver frees what it made", "[solvers][bitwuzla]")
{
  require_no_growth(optionst());
}

TEST_CASE(
  "a Bitwuzla solver frees the flatteners built for it",
  "[solvers][bitwuzla]")
{
  optionst options;
  options.set_option("array-flattener", true);
  options.set_option("fp2bv", true);
  options.set_option("tuple-sym-flattener", true);
  require_no_growth(options);
}

TEST_CASE(
  "a Bitwuzla solver frees the sort a zeroed struct table needs",
  "[solvers][bitwuzla]")
{
  require_no_growth(optionst(), solve_zeroed_table_once);
}
