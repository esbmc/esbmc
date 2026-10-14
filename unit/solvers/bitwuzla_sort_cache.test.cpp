// Sort and model-value ownership in the Bitwuzla backend.
//
// Bitwuzla's C API refcounts every sort and term it exports and expects the
// caller to release them (src/api/c/bitwuzla_structs.cpp). The backend used to
// mint a fresh solver_smt_sort -- and with it a fresh Bitwuzla sort reference
// -- on every mk_bv_sort call, and mk_extract/mk_concat/mk_sign_ext/mk_zero_ext
// ask for one per operation, so neither was ever freed. The first SCENARIO
// pins the cache that fixes it: a sort is built once per distinct sort, and
// the key discriminates the kinds that share a width.
//
// The second drives a full solve and reads every value back, which is the only
// way the matching model-query fix is observable: over-releasing the reference
// bitwuzla_get_value() hands out is a use-after-free inside the term manager,
// not a wrong answer.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <memory>
#include <irep2/irep2_utils.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_result.h>
#include <solvers/smt/smt_solver.h>
#include <solvers/solve.h>
#include <util/arith/arith_tools.h>
#include <util/config/config.h>
#include <util/config/options.h>
#include <util/lang/c_types.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>
#include <bitwuzla_conv.h>

// The Bitwuzla backend publishes no header for its creator; solve.cpp declares
// the type, bitwuzla_conv.cpp defines the symbol.
extern solver_creator create_new_bitwuzla_solver;

SCENARIO("the Bitwuzla backend builds each sort once", "[solvers][bitwuzla]")
{
  contextt ctx;
  namespacet ns(ctx);
  optionst options;
  tuple_iface *tuple_api = nullptr;
  array_iface *array_api = nullptr;
  fp_convt *fp_api = nullptr;
  std::unique_ptr<smt_solver_baset> base{
    create_new_bitwuzla_solver(options, ns, &tuple_api, &array_api, &fp_api)};
  REQUIRE(base != nullptr);
  bitwuzla_convt *solver = dynamic_cast<bitwuzla_convt *>(base.get());
  REQUIRE(solver != nullptr);

  GIVEN("a bit-vector sort requested twice at the same width")
  {
    THEN("the same sort object comes back")
    {
      REQUIRE(solver->mk_bv_sort(32) == solver->mk_bv_sort(32));
    }
    THEN("a different width is a different sort")
    {
      REQUIRE(solver->mk_bv_sort(32) != solver->mk_bv_sort(64));
    }
    THEN("the cached sort still reports its own width")
    {
      REQUIRE(solver->mk_bv_sort(32)->get_data_width() == 32);
      REQUIRE(solver->mk_bv_sort(64)->get_data_width() == 64);
    }
  }

  GIVEN("two sort kinds that share a width")
  {
    THEN("a fixed-point sort is not the bit-vector sort of the same width")
    {
      REQUIRE(solver->mk_bv_sort(32) != solver->mk_fbv_sort(32));
      REQUIRE(solver->mk_fbv_sort(32)->id == SMT_SORT_FIXEDBV);
    }
    THEN("a bit-vector float is not the native float of the same format")
    {
      REQUIRE(solver->mk_bvfp_sort(8, 23) != solver->mk_fpbv_sort(8, 23));
      REQUIRE(solver->mk_bvfp_sort(8, 23)->id == SMT_SORT_BVFP);
      REQUIRE(solver->mk_fpbv_sort(8, 23)->id == SMT_SORT_FPBV);
    }
    THEN("the two rounding-mode sorts stay distinct")
    {
      REQUIRE(solver->mk_bvfp_rm_sort() != solver->mk_fpbv_rm_sort());
    }
  }

  GIVEN("the nullary sorts")
  {
    THEN("each is built once")
    {
      REQUIRE(solver->mk_bool_sort() == solver->mk_bool_sort());
      REQUIRE(solver->mk_bvfp_rm_sort() == solver->mk_bvfp_rm_sort());
      REQUIRE(solver->mk_fpbv_rm_sort() == solver->mk_fpbv_rm_sort());
    }
  }

  GIVEN("floating-point sorts differing in their field widths")
  {
    // Bitwuzla is built --no-fpexp, so a native fp sort must name one of the
    // standard formats; the bit-vector encodings below are unconstrained.
    THEN("a native float sort is built once per format")
    {
      REQUIRE(solver->mk_fpbv_sort(8, 23) == solver->mk_fpbv_sort(8, 23));
      REQUIRE(solver->mk_fpbv_sort(8, 23) != solver->mk_fpbv_sort(11, 52));
      REQUIRE(solver->mk_fpbv_sort(5, 10) != solver->mk_fpbv_sort(8, 23));
    }
    THEN("exponent and significand discriminate independently")
    {
      // Same total width, different split: a key on width alone collides.
      REQUIRE(solver->mk_bvfp_sort(8, 23) != solver->mk_bvfp_sort(11, 20));
      REQUIRE(
        solver->mk_bvfp_sort(8, 23)->get_data_width() ==
        solver->mk_bvfp_sort(11, 20)->get_data_width());
      // Differing only in the significand, so the key's last field carries it.
      REQUIRE(solver->mk_bvfp_sort(8, 23) != solver->mk_bvfp_sort(8, 24));
      REQUIRE(solver->mk_bvfp_sort(8, 24)->get_data_width() == 33);
    }
  }

  GIVEN("array sorts over cached domains and ranges")
  {
    smt_sortt dom = solver->mk_bv_sort(8);
    smt_sortt u32 = solver->mk_bv_sort(32);
    smt_sortt u64 = solver->mk_bv_sort(64);

    THEN("the same domain and range give the same array sort")
    {
      REQUIRE(
        solver->mk_array_sort(dom, u32) == solver->mk_array_sort(dom, u32));
    }
    THEN("a different range is a different array sort")
    {
      REQUIRE(
        solver->mk_array_sort(dom, u32) != solver->mk_array_sort(dom, u64));
    }
    THEN("a different domain is a different array sort")
    {
      REQUIRE(
        solver->mk_array_sort(dom, u32) !=
        solver->mk_array_sort(solver->mk_bv_sort(16), u32));
      // Same width, different kind: the domain half of the key is an identity,
      // not a width, so these must not collide.
      REQUIRE(
        solver->mk_array_sort(solver->mk_bv_sort(8), u32) !=
        solver->mk_array_sort(solver->mk_fbv_sort(8), u32));
    }
    THEN("the cached array sort keeps its domain width and range")
    {
      smt_sortt arr = solver->mk_array_sort(dom, u32);
      REQUIRE(arr->id == SMT_SORT_ARRAY);
      REQUIRE(arr->get_domain_width() == 8);
      REQUIRE(arr->get_range_sort() == u32);
    }
  }
}

SCENARIO("Bitwuzla model values survive being read back", "[solvers][bitwuzla]")
{
  config.ansi_c.set_data_model(configt::LP64);
  contextt ctx;
  namespacet ns(ctx);
  optionst options;
  std::unique_ptr<smt_convt> solver{create_solver("bitwuzla", ns, options)};
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

  // create_new_bitwuzla_solver hands the convt out as the fp_convt, so a
  // floatbv symbol routes smt_convt::get() into bitwuzla_convt::get_fpbv.
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
