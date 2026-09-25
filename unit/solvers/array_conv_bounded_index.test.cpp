// Boundary contract for array_convt's bounded select/store (issue #6638).
//
// `array_ast::array_fields` is indexed 0..size()-1, so a constant index equal
// to size() is already out of range. Both paths guarded with `>` instead of
// `>=`, letting exactly that index through to `array_fields[size()]` -- a read
// in mk_select and a write in mk_store, i.e. one past the end of a
// std::vector.
//
// The branch is not reachable from C input: constant-index accesses are folded
// before conversion, so every array that reaches the flattener in practice
// either has a symbolic index or a 64-bit domain (making it unbounded). This
// test therefore drives the two entry points directly. Under a sanitizer the
// pre-fix code reports a container-overflow here; without one, the store case
// is still discriminating because the guard returns the *original* ast whereas
// the out-of-range write returns a fresh copy.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <memory>
#include <util/config/options.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>
#include <irep2/irep2_utils.h>
#include <solvers/smt/smt_solver.h>
#include <solvers/smt/array_conv.h>
#include <solvers/solve.h>

extern solver_creator create_new_z3_solver;

SCENARIO(
  "array_convt rejects a constant index equal to the bounded array's size",
  "[solvers][array]")
{
  contextt ctx;
  namespacet ns(ctx);
  optionst options;
  tuple_iface *tuple_api = nullptr;
  array_iface *array_api = nullptr;
  fp_convt *fp_api = nullptr;
  std::unique_ptr<smt_solver_baset> solver{
    create_new_z3_solver(options, ns, &tuple_api, &array_api, &fp_api)};
  REQUIRE(solver != nullptr);

  array_convt flattener(solver.get());

  GIVEN("a bounded array of four elements")
  {
    // A domain width of 2 keeps is_unbounded_array() false (it requires <= 10)
    // and gives mk_array_symbol exactly 1 << 2 == 4 fields.
    smt_sortt subtype = solver->mk_int_bv_sort(8);
    smt_sortt domain = solver->mk_int_bv_sort(2);
    smt_sortt arrsort = solver->mk_array_sort(domain, subtype);

    smt_astt arr = flattener.mk_array_symbol("bounded_array", arrsort, subtype);
    const array_ast *ma = array_downcast(arr);
    REQUIRE(ma->array_fields.size() == 4);

    const type2tc idx_type = get_uint_type(2);

    WHEN("selecting at the last valid index")
    {
      expr2tc in_range = constant_int2tc(idx_type, BigInt(3));
      THEN("the stored field is returned")
      {
        REQUIRE(
          flattener.mk_select(ma, in_range, subtype) == ma->array_fields[3]);
      }
    }

    WHEN("selecting at an index equal to the size")
    {
      expr2tc out_of_range = constant_int2tc(idx_type, BigInt(4));
      THEN("a fresh value is returned rather than a field past the end")
      {
        smt_astt got = flattener.mk_select(ma, out_of_range, subtype);
        REQUIRE(got != nullptr);
        for (smt_astt field : ma->array_fields)
          REQUIRE(got != field);
      }
    }

    WHEN("storing at an index equal to the size")
    {
      expr2tc out_of_range = constant_int2tc(idx_type, BigInt(4));
      smt_astt value = solver->mk_smt_bv(BigInt(7), subtype);
      THEN("the array is returned unchanged rather than written past the end")
      {
        REQUIRE(flattener.mk_store(ma, out_of_range, value, arrsort) == arr);
      }
    }
  }
}
