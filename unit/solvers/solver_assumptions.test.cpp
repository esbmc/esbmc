// Solving under assumptions and reading back which ones an UNSAT result used.
// The transition-system engines key every query on activation literals, so the
// assumptions must hold for one query only and survive push/pop around them.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <memory>
#include <irep2/irep2_utils.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_result.h>
#include <solvers/solve.h>
#include <util/arith/arith_tools.h>
#include <util/config/config.h>
#include <util/config/options.h>
#include <util/lang/c_types.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>

SCENARIO("assumptions hold for one query", "[solvers][bitwuzla]")
{
  config.ansi_c.set_data_model(configt::LP64);
  contextt ctx;
  namespacet ns(ctx);
  optionst options;
  options.set_option("smt-unsat-assumptions", true);
  std::unique_ptr<smt_convt> solver{create_solver("bitwuzla", ns, options)};
  REQUIRE(solver->supports_assumptions());

  const type2tc u8 = get_uint8_type();
  expr2tc a = symbol2tc(get_bool_type(), "a");
  expr2tc b = symbol2tc(get_bool_type(), "b");
  expr2tc x = symbol2tc(u8, "x");
  solver->assert_expr(implies2tc(a, equality2tc(x, from_integer(1, u8))));
  solver->assert_expr(implies2tc(b, equality2tc(x, from_integer(2, u8))));

  auto value_of_x = [&]() {
    return to_constant_int2t(solver->get(x)).value.to_uint64();
  };

  REQUIRE(solver->dec_solve_assuming({a}) == smt_resultt::P_SATISFIABLE);
  REQUIRE(value_of_x() == 1);

  REQUIRE(solver->dec_solve_assuming({a, b}) == smt_resultt::P_UNSATISFIABLE);
  std::vector<expr2tc> core = solver->unsat_assumptions();
  REQUIRE(!core.empty());
  for (const expr2tc &e : core)
    REQUIRE((e == a || e == b));

  REQUIRE(solver->dec_solve_assuming({b}) == smt_resultt::P_SATISFIABLE);
  REQUIRE(value_of_x() == 2);

  solver->assert_expr(not2tc(b));
  REQUIRE(solver->dec_solve_assuming({b}) == smt_resultt::P_UNSATISFIABLE);
  core = solver->unsat_assumptions();
  REQUIRE(core.size() == 1);
  REQUIRE(core[0] == b);

  solver->push_ctx();
  solver->assert_expr(not2tc(a));
  REQUIRE(solver->dec_solve_assuming({a}) == smt_resultt::P_UNSATISFIABLE);
  solver->pop_ctx();
  REQUIRE(solver->dec_solve_assuming({a}) == smt_resultt::P_SATISFIABLE);
  REQUIRE(value_of_x() == 1);
}
