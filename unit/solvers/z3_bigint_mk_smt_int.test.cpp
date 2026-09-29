// Precision contract for the Z3 backend's mk_smt_int on out-of-int64
// BigInts (issue #4642).
//
// Pre-camada this lived in z3_convt::mk_smt_int and the fix routed values
// outside the int64 range through Z3's string overload. The camada port
// preserves the contract: src/solvers/camada/camada_conv.cpp::mk_smt_int
// hands camada the decimal string via integer2string. A regression that
// reintroduced a to_int64 / is_uint64 fast path on either side of that
// boundary would corrupt the constant before the solver ever sees it.
//
// We exercise ESBMC's overload end-to-end: assert a fresh symbol equals the
// constant, solve, and read the value back as a BigInt via get_bv. Testing
// Z3's own numeral API is camada's responsibility now, not ours.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <memory>
#include <big-int/bigint.hh>
#include <util/symtab/context.h>
#include <util/arith/mp_arith.h>
#include <util/symtab/namespace.h>
#include <util/config/options.h>
#include <solvers/smt_solver.h>
#include <solvers/solve.h>

SCENARIO(
  "the Z3 backend's mk_smt_int preserves BigInt precision and sign",
  "[z3][bigint]")
{
  // Drive ESBMC's overload through camada's Z3 backend so a regression that
  // reintroduces a to_int64 / is_uint64 fast path in either mk_smt_int or
  // integer2string surfaces here, not at the next bignum-aware caller.
  //
  // The AST-level API (mk_smt_int / mk_eq / get_bv) lives on
  // smt_solver_baset and is deliberately unreachable through smt_conv.h, so
  // go through the creator directly rather than create_solver().
  contextt ctx;
  namespacet ns(ctx);
  optionst options;
  // mk_smt_int returns an Int sort, which only works in integer-encoding
  // mode. Without this the solver would reject the assert below.
  options.set_option("int-encoding", true);
  std::unique_ptr<smt_solver_baset> solver =
    create_linked_solver(options, ns, create_esbmc_z3_solver);
  REQUIRE(solver != nullptr);

  // create_solver() does this wiring plus smt_post_init(); replicate the part
  // mk_eq / assert_ast need (boolean_sort and the address-space scaffolding).
  solver->smt_post_init();

  // Round-trip a BigInt through the solver: assert x == mk_smt_int(v), solve,
  // read x back via get_bv. If mk_smt_int (or anything beneath it) drops bits
  // or sign, the readback won't match the input.
  auto roundtrip = [&](const BigInt &v) {
    smt_astt sym = solver->mk_smt_symbol("test_bigint", solver->mk_int_sort());
    smt_astt eq = solver->mk_eq(sym, solver->mk_smt_int(v));
    solver->assert_ast(eq);
    REQUIRE(solver->dec_solve() == P_SATISFIABLE);
    return solver->get_bv(sym, /*is_signed=*/true);
  };

  GIVEN("A BigInt that fits int64")
  {
    BigInt v(-12345);
    THEN("mk_smt_int round-trips through the solver")
    {
      REQUIRE(roundtrip(v) == v);
    }
  }

  GIVEN("A BigInt larger than uint64_max")
  {
    BigInt huge = BigInt::power2(200);
    THEN("mk_smt_int preserves every digit via the string path")
    {
      REQUIRE_FALSE(huge.is_int64());
      REQUIRE(roundtrip(huge) == huge);
    }
  }

  GIVEN("A negative BigInt with magnitude in (INT64_MAX, UINT64_MAX]")
  {
    // The regression this case pins: a previous draft of the fix included
    // an `is_uint64` fast-path that silently flipped this value's sign.
    BigInt edge = -(BigInt::power2(63) + BigInt(1));
    THEN("mk_smt_int preserves the sign")
    {
      REQUIRE_FALSE(edge.is_int64());
      REQUIRE(edge.is_uint64());
      REQUIRE(roundtrip(edge) == edge);
    }
  }

  GIVEN("A negative BigInt smaller than int64_min")
  {
    BigInt neg = -BigInt::power2(200);
    THEN("mk_smt_int preserves the sign and every digit")
    {
      REQUIRE_FALSE(neg.is_int64());
      REQUIRE(roundtrip(neg) == neg);
    }
  }
}
