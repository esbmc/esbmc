// Semantics of smt_solver_baset::mk_guarded_choice (esbmc/esbmc#82): an N-way
// guarded selection encoded as a free variable plus flat implications instead
// of a nested ite chain.
//
// array_convt::mk_select, the only in-tree caller, always passes a null
// default because an index with no matching field must stay nondeterministic.
// The pinned-default arm is therefore unreachable from C input and only a
// direct call can state its contract.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <memory>
#include <util/config/config.h>
#include <util/config/options.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>
#include <solvers/smt/smt_solver.h>
#include <solvers/solve.h>

extern solver_creator create_new_z3_solver;

SCENARIO("mk_guarded_choice constrains a free variable by implication",
         "[solvers][guarded-choice]")
{
  config.ansi_c.set_data_model(configt::LP64);
  contextt ctx;
  namespacet ns(ctx);
  optionst options;
  tuple_iface *tuple_api = nullptr;
  array_iface *array_api = nullptr;
  fp_convt *fp_api = nullptr;
  std::unique_ptr<smt_solver_baset> solver{
    create_new_z3_solver(options, ns, &tuple_api, &array_api, &fp_api)};
  REQUIRE(solver != nullptr);

  // create_solver() does this wiring; dec_solve() and the boolean sort used by
  // eq()/mk_not() are not usable before smt_post_init().
  REQUIRE(tuple_api != nullptr);
  REQUIRE(array_api != nullptr);
  REQUIRE(fp_api != nullptr);
  solver->set_tuple_iface(tuple_api);
  solver->set_array_iface(array_api);
  solver->set_fp_conv(fp_api);
  solver->smt_post_init();

  GIVEN("a two-way choice on an 8-bit selector")
  {
    smt_sortt s8 = solver->mk_int_bv_sort(8);
    smt_astt k = solver->mk_smt_symbol("k", s8);
    smt_astt v0 = solver->mk_smt_bv(BigInt(10), s8);
    smt_astt v1 = solver->mk_smt_bv(BigInt(20), s8);
    smt_astt fallback = solver->mk_smt_bv(BigInt(99), s8);

    std::vector<std::pair<smt_astt, smt_astt>> cases{
      {k->eq(solver.get(), solver->mk_smt_bv(BigInt(0), s8)), v0},
      {k->eq(solver.get(), solver->mk_smt_bv(BigInt(1), s8)), v1}};

    WHEN("a guard holds")
    {
      smt_astt out =
        solver->mk_guarded_choice(s8, "gc_hit::", cases, nullptr);
      solver->assert_ast(k->eq(solver.get(), solver->mk_smt_bv(BigInt(1), s8)));
      solver->assert_ast(solver->mk_not(out->eq(solver.get(), v1)));

      THEN("the result is pinned to that case's value")
      {
        REQUIRE(solver->dec_solve() == P_UNSATISFIABLE);
      }
    }

    WHEN("no guard holds and there is no default")
    {
      smt_astt out =
        solver->mk_guarded_choice(s8, "gc_nondet::", cases, nullptr);
      solver->assert_ast(k->eq(solver.get(), solver->mk_smt_bv(BigInt(7), s8)));
      solver->assert_ast(solver->mk_not(out->eq(solver.get(), v0)));
      solver->assert_ast(solver->mk_not(out->eq(solver.get(), v1)));

      THEN("the result is unconstrained")
      {
        REQUIRE(solver->dec_solve() == P_SATISFIABLE);
      }
    }

    WHEN("no guard holds and a default is supplied")
    {
      smt_astt out =
        solver->mk_guarded_choice(s8, "gc_default::", cases, fallback);
      solver->assert_ast(k->eq(solver.get(), solver->mk_smt_bv(BigInt(7), s8)));
      solver->assert_ast(solver->mk_not(out->eq(solver.get(), fallback)));

      THEN("the result is pinned to the default")
      {
        REQUIRE(solver->dec_solve() == P_UNSATISFIABLE);
      }
    }

    WHEN("a guard holds and a default is supplied")
    {
      smt_astt out =
        solver->mk_guarded_choice(s8, "gc_both::", cases, fallback);
      solver->assert_ast(k->eq(solver.get(), solver->mk_smt_bv(BigInt(0), s8)));
      solver->assert_ast(solver->mk_not(out->eq(solver.get(), v0)));

      THEN("the guard wins over the default")
      {
        REQUIRE(solver->dec_solve() == P_UNSATISFIABLE);
      }
    }
  }
}
