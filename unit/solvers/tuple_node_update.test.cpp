// Field-preservation contract for a struct update: `with(s, a, 5)` must agree
// with `s` on every field but `a`.
//
// Written against ESBMC's own tuple node flattener, whose update() copied the
// source's `elements` before the source had been made free -- an unpopulated
// source copied an empty vector and make_free() then invented fresh
// unconstrained variables for every field. That flattener is gone: camada
// implements tuples itself (solve.cpp). The contract outlives it, so this
// keeps checking it against whatever the backend does.
//
// Not reachable from C input, which is why the original bug went unnoticed:
// the simplifier folds member-over-with away, and a source tuple is normally
// populated by its own defining assignment before any update reaches it. Only
// a query holding both the updated and the original tuple at once -- as the
// simplifier equivalence check does -- puts the two side by side.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <memory>
#include <irep2/irep2_utils.h>
#include <solvers/smt_conv.h>
#include <solvers/smt_result.h>
#include <solvers/solve.h>
#include <util/arith/arith_tools.h>
#include <util/config/config.h>
#include <util/config/options.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>

SCENARIO(
  "a struct update leaves the fields it does not touch alone",
  "[solvers][tuple]")
{
  config.ansi_c.set_data_model(configt::LP64);
  contextt ctx;
  namespacet ns(ctx);
  optionst options;

  const type2tc int32 = get_int32_type();
  const type2tc s_type = struct_type2tc(
    std::vector<type2tc>{int32, int32},
    std::vector<irep_idt>{"a", "b"},
    std::vector<irep_idt>{"a", "b"},
    "S",
    false);

  const expr2tc s = symbol2tc(s_type, "s");
  const expr2tc field_a = constant_string2tc(
    array_type2tc(get_uint8_type(), from_integer(2, size_type2()), false),
    "a",
    constant_string_kindt::DEFAULT);
  const expr2tc updated = with2tc(s_type, s, field_a, from_integer(5, int32));

  GIVEN("a solver holding both the updated struct and the original")
  {
    std::unique_ptr<smt_convt> solver{create_solver(ns, options)};
    REQUIRE(solver != nullptr);

    THEN("the untouched field reads the same through either")
    {
      solver->assert_expr(
        notequal2tc(member2tc(int32, updated, "b"), member2tc(int32, s, "b")));
      REQUIRE(solver->dec_solve() == P_UNSATISFIABLE);
    }
  }
}
