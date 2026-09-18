// check_solver_availability() rejects a solver name no build carries. It used
// to abort(), which dumped core and exited 134 — read by CI as a crash in the
// verifier rather than a rejected input (esbmc/esbmc#7901). The regression
// tests pin the exit status end to end; this pins the mechanism, and stays
// meaningful whichever backends the build compiled in.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <solvers/solve.h>
#include <util/base/user_input_error.h>
#include <util/config/options.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>

TEST_CASE("an unbuilt solver is rejected, not aborted", "[solvers]")
{
  optionst options;
  options.set_option("default-solver", "nosuchsolver");
  REQUIRE_THROWS_AS(check_solver_availability(options), user_input_errort);
  REQUIRE_THROWS_WITH(check_solver_availability(options), "input rejected");
}

// The CLI validates the choice first, so only a direct caller reaches this.
TEST_CASE("create_solver rejects an unbuilt solver name", "[solvers]")
{
  optionst options;
  contextt context;
  namespacet ns(context);
  REQUIRE_THROWS_AS(
    create_solver("nosuchsolver", ns, options), user_input_errort);
}

TEST_CASE("two solvers at once are rejected, not aborted", "[solvers]")
{
  optionst options;
  options.set_option("z3", true);
  options.set_option("boolector", true);
  REQUIRE_THROWS_AS(check_solver_availability(options), user_input_errort);
}

TEST_CASE("no solver named is not an error", "[solvers]")
{
  optionst options;
  options.set_option("default-solver", "");
  REQUIRE_NOTHROW(check_solver_availability(options));
}
