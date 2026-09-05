// Model-value ownership, driven through smt_convt.
//
// Bitwuzla's C API refcounts every term it exports and expects the caller to
// release them. Upstream's companion scenario pinned ESBMC's own sort cache by
// constructing bitwuzla_convt directly; camada owns that lifetime now, so only
// this half ports. It drives a full solve and reads every value back, which is
// the only way a model-query ownership bug is observable: over-releasing the
// reference the value query hands out is a use-after-free inside the term
// manager, not a wrong answer.
//
// Backend-agnostic on purpose -- create_solver picks whichever camada was
// built with, so this runs wherever the suite does.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <memory>
#include <irep2/irep2_utils.h>
#include <solvers/smt_conv.h>
#include <solvers/smt_result.h>
#include <solvers/smt_solver.h>
#include <solvers/solve.h>
#include <util/arith/arith_tools.h>
#include <util/config/config.h>
#include <util/config/options.h>
#include <util/lang/c_types.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>

SCENARIO("model values survive being read back", "[solvers]")
{
  config.ansi_c.set_data_model(configt::LP64);
  contextt ctx;
  namespacet ns(ctx);
  optionst options;
  std::unique_ptr<smt_convt> solver{create_solver(ns, options)};
  REQUIRE(solver != nullptr);

  const type2tc u32 = get_uint32_type();
  const type2tc arr_type = array_type2tc(u32, from_integer(4, u32), false);

  // One constrained scalar per query, plus a bool, a float and an array
  // element, so every backend entry point that takes a value_reft runs at
  // least once: get_bv, get_bool, get_fpbv, get_array_elem and print_model.
  std::vector<expr2tc> scalars;
  for (unsigned i = 0; i < 8; i++)
  {
    expr2tc sym = symbol2tc(u32, "s" + std::to_string(i));
    solver->assert_expr(equality2tc(sym, from_integer(0xdead0000 + i, u32)));
    scalars.push_back(sym);
  }

  const expr2tc flag = symbol2tc(get_bool_type(), "flag");
  solver->assert_expr(flag);

  const expr2tc arr = symbol2tc(arr_type, "a");
  const expr2tc elem = index2tc(u32, arr, from_integer(2, u32));
  solver->assert_expr(equality2tc(elem, from_integer(0xbeef, u32)));

  // A floatbv symbol routes smt_convt::get() through the backend's own
  // floating-point value query.
  ieee_floatt half(ieee_float_spect::double_precision());
  half.from_double(0.5);
  const expr2tc fsym =
    symbol2tc(ieee_float_spect::double_precision().get_type(), "f");
  solver->assert_expr(equality2tc(fsym, constant_floatbv2tc(half)));

  REQUIRE(solver->dec_solve() == smt_resultt::P_SATISFIABLE);

  GIVEN("a satisfiable model")
  {
    THEN("every scalar reads back its asserted value")
    {
      for (unsigned i = 0; i < scalars.size(); i++)
      {
        expr2tc v = solver->get(scalars[i]);
        REQUIRE(is_constant_int2t(v));
        REQUIRE(to_constant_int2t(v).value == BigInt(0xdead0000 + i));
      }
    }
    THEN("the boolean reads back true")
    {
      REQUIRE(solver->l_get(flag).is_true());
    }
    THEN("the array element reads back its asserted value")
    {
      expr2tc v = solver->get(elem);
      REQUIRE(is_constant_int2t(v));
      REQUIRE(to_constant_int2t(v).value == BigInt(0xbeef));
    }
    THEN("the float reads back its asserted value")
    {
      expr2tc v = solver->get(fsym);
      REQUIRE(is_constant_floatbv2t(v));
      REQUIRE(to_constant_floatbv2t(v).value.to_double() == 0.5);
    }
    THEN("printing the model releases every value it reads")
    {
      solver->print_model();
    }
    THEN("repeating every query returns the same values")
    {
      for (unsigned round = 0; round < 3; round++)
      {
        for (unsigned i = 0; i < scalars.size(); i++)
          REQUIRE(
            to_constant_int2t(solver->get(scalars[i])).value ==
            BigInt(0xdead0000 + i));
        REQUIRE(solver->l_get(flag).is_true());
        REQUIRE(to_constant_int2t(solver->get(elem)).value == BigInt(0xbeef));
      }
    }
  }
}
