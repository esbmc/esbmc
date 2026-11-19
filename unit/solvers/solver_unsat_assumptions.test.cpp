// Unsat assumptions. PDR generalises a blocked cube to the literals an UNSAT
// query needed, so every backend must report the assumptions the proof used
// and leave out the ones it did not.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <memory>
#include <irep2/irep2_utils.h>
#include <solver_config.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_result.h>
#include <solvers/solve.h>
#include <util/arith/arith_tools.h>
#include <util/config/config.h>
#include <util/config/options.h>
#include <util/lang/c_types.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>

static const char *const backends[] = {
#ifdef Z3
  "z3",
#endif
#ifdef BITWUZLA
  "bitwuzla",
#endif
#ifdef BOOLECTOR
  "boolector",
#endif
#ifdef USECVC5
  "cvc5",
#endif
#ifdef YICES
  "yices",
#endif
};

SCENARIO("unsat assumptions name what the proof used", "[solvers]")
{
  config.ansi_c.set_data_model(configt::LP64);
  for (const std::string name : backends)
  {
    SECTION(name)
    {
      contextt ctx;
      namespacet ns(ctx);
      optionst options;
      options.set_option("smt-assumptions", true);
      options.set_option("smt-unsat-assumptions", true);
      std::unique_ptr<smt_convt> solver{create_solver(name, ns, options)};

      const type2tc u8 = get_uint8_type();
      expr2tc a = symbol2tc(get_bool_type(), "a");
      expr2tc b = symbol2tc(get_bool_type(), "b");
      expr2tc c = symbol2tc(get_bool_type(), "c");
      expr2tc x = symbol2tc(u8, "x");
      solver->assert_expr(implies2tc(a, equality2tc(x, from_integer(1, u8))));
      solver->assert_expr(implies2tc(b, equality2tc(x, from_integer(2, u8))));

      REQUIRE(
        solver->dec_solve_assuming({c, a, b}) == smt_resultt::P_UNSATISFIABLE);
      std::vector<expr2tc> core = solver->unsat_assumptions();
      REQUIRE(core.size() == 2);
      REQUIRE(std::count(core.begin(), core.end(), a) == 1);
      REQUIRE(std::count(core.begin(), core.end(), b) == 1);

      // A later query reports its own core, not the last one.
      solver->assert_expr(not2tc(c));
      REQUIRE(solver->dec_solve_assuming({c}) == smt_resultt::P_UNSATISFIABLE);
      REQUIRE(solver->unsat_assumptions() == std::vector<expr2tc>{c});
    }
  }
}
