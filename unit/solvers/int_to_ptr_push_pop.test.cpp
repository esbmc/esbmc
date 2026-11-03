// A pointer made from an integer names an object through the base
// address-space version (#7855). The base is tied to the objects' ranges on
// first use, so a pop must forget a tie made inside the popped frame, or the
// next frame reuses a tie the solver no longer holds.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <memory>
#include <irep2/irep2_utils.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_result.h>
#include <solvers/solve.h>
#include <util/config/config.h>
#include <util/config/options.h>
#include <util/lang/c_types.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>

SCENARIO(
  "a pop forgets the base address space tied inside it",
  "[solvers][bitwuzla]")
{
  config.ansi_c.set_data_model(configt::LP64);
  contextt ctx;
  namespacet ns(ctx);
  optionst options;
  std::unique_ptr<smt_convt> solver{create_solver("bitwuzla", ns, options)};

  const type2tc int_ptr = pointer_type2tc(get_int32_type());
  const type2tc uintptr = get_uint64_type();
  const expr2tc g =
    address_of2tc(get_int32_type(), symbol2tc(get_int32_type(), "g"));
  solver->assert_expr(notequal2tc(g, gen_zero(int_ptr)));

  const expr2tc u = symbol2tc(uintptr, "u");
  const expr2tc names_g_elsewhere = and2tc(
    equality2tc(typecast2tc(int_ptr, u), g),
    notequal2tc(u, typecast2tc(uintptr, g)));

  for (int frame = 0; frame < 2; ++frame)
  {
    solver->push_ctx();
    solver->assert_expr(names_g_elsewhere);
    REQUIRE(solver->dec_solve() == smt_resultt::P_UNSATISFIABLE);
    solver->pop_ctx();
  }
}
