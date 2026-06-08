// Round-trip tests for the IREP1 <-> IREP2 conversion layer (util/migrate.cpp).
//
// These pin the foundational property the goto-program IREP2 migration relies
// on (esbmc/esbmc#4715): the IREP2 representation is stable under a round-trip
// through the legacy irept form. Concretely, for every type/expression kind a
// symbol can hold,
//
//     migrate_type(migrate_type_back(T2)) == T2
//     migrate_expr(migrate_expr_back(E2)) == E2
//
// i.e. back-migrating an IREP2 node to legacy and re-migrating yields an equal
// IREP2 node. The symbol-table migration (Phase 4) derives the legacy
// symbolt::type/value from IREP2 shadow fields via migrate_*_back; this
// idempotence is what makes that derivation lossless. We assert IREP2 round-trip
// (not legacy byte-equality) because migrate_type/expr deliberately canonicalise
// some legacy forms, so the IREP2 side is the stable reference.

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2.h>
#include <irep2/irep2_utils.h>
#include <util/migrate.h>
#include <util/namespace.h>
#include <util/context.h>
#include <util/expr_util.h>

namespace
{
// Migrating a legacy symbol_exprt resolves the symbol's type through the
// global migrate_namespace_lookup (migrate.cpp). Register a namespace with the
// symbols our expr tests reference, and point the global at it. This mirrors
// what language_ui does in the real pipeline.
contextt &test_context()
{
  static contextt ctx;
  static bool initialised = false;
  if (!initialised)
  {
    symbolt x;
    x.id = x.name = "x";
    x.set_type(migrate_type_back(get_int_type(32)));
    x.lvalue = true;
    ctx.add(x);
    initialised = true;
  }
  return ctx;
}

// Point the migrate layer at our namespace. Done at call time (not via a
// static initialiser) to avoid the cross-translation-unit static-init order
// fiasco with migrate.cpp's own `migrate_namespace_lookup = nullptr`.
void use_test_ns()
{
  static namespacet ns(test_context());
  migrate_namespace_lookup = &ns;
}

void require_type_roundtrip(const type2tc &t2)
{
  type2tc back = migrate_type(migrate_type_back(t2));
  INFO("type kind id = " << get_type_id(t2));
  REQUIRE(back == t2);
}

void require_expr_roundtrip(const expr2tc &e2)
{
  expr2tc back;
  migrate_expr(migrate_expr_back(e2), back);
  INFO("expr kind id = " << get_expr_id(e2));
  REQUIRE(back == e2);
}

type2tc make_func_type()
{
  // int f(int a, unsigned b)
  std::vector<type2tc> args{get_int_type(32), get_uint_type(32)};
  std::vector<irep_idt> names{"a", "b"};
  return code_type2tc(args, get_int_type(32), names, /*ellipsis=*/false);
}

type2tc make_struct_type()
{
  std::vector<type2tc> members{get_int_type(32), get_bool_type()};
  std::vector<irep_idt> names{"x", "y"};
  return struct_type2tc(members, names, names, "s");
}
} // namespace

TEST_CASE("migrate type round-trips for scalar kinds", "[migrate]")
{
  require_type_roundtrip(get_bool_type());
  require_type_roundtrip(get_int_type(32));
  require_type_roundtrip(get_uint_type(64));
  require_type_roundtrip(get_empty_type());
}

TEST_CASE("migrate type round-trips for composite kinds", "[migrate]")
{
  require_type_roundtrip(pointer_type2tc(get_int_type(32)));
  // fixed-size and infinite-size arrays. Use an explicit-width size constant:
  // gen_ulong() would derive its width from config.ansi_c, which is not set up
  // in a bare unit test (yielding a 0-width size that cannot hold the value).
  require_type_roundtrip(array_type2tc(
    get_int_type(32),
    constant_int2tc(get_uint_type(64), BigInt(4)),
    /*inf=*/false));
  require_type_roundtrip(array_type2tc(get_uint_type(8), expr2tc(), true));
  require_type_roundtrip(make_struct_type());
}

TEST_CASE("migrate type round-trips for function signatures", "[migrate]")
{
  // The function-signature kind that goto_functiont::type holds (Phase 3).
  require_type_roundtrip(make_func_type());
  // void g(void)
  require_type_roundtrip(code_type2tc(
    std::vector<type2tc>{},
    get_empty_type(),
    std::vector<irep_idt>{},
    /*ellipsis=*/false));
}

TEST_CASE("migrate expr round-trips for constant kinds", "[migrate]")
{
  require_expr_roundtrip(constant_int2tc(get_int_type(32), BigInt(42)));
  require_expr_roundtrip(gen_true_expr());
  require_expr_roundtrip(gen_false_expr());
  std::vector<expr2tc> members{
    constant_int2tc(get_int_type(32), BigInt(1)), gen_true_expr()};
  require_expr_roundtrip(constant_struct2tc(make_struct_type(), members));
}

TEST_CASE("migrate expr round-trips for symbol and address-of", "[migrate]")
{
  use_test_ns();
  expr2tc sym = symbol2tc(get_int_type(32), "x");
  require_expr_roundtrip(sym);
  require_expr_roundtrip(address_of2tc(get_int_type(32), sym));
}

TEST_CASE("migrate expr round-trips for code statements", "[migrate]")
{
  // Individual code statements round-trip (goto2c back-migrates instructions
  // one at a time).
  use_test_ns();
  expr2tc lhs = symbol2tc(get_int_type(32), "x");
  expr2tc rhs = constant_int2tc(get_int_type(32), BigInt(7));
  require_expr_roundtrip(code_assign2tc(lhs, rhs));
}

// V1 of the symbol-table V-track (esbmc/esbmc#4715): five expr2t kinds had
// gaps in the migration layer (no back-arm, or no forward-arm, or neither).
// These tests pin the round-trip property -- migrate_expr_back followed by
// migrate_expr returns an equal IREP2 node -- that the value-side flip (V2)
// relies on. The arms are dead code in the pipeline today; nothing constructs
// the matching legacy form. The tests exercise them via the IREP2 constructors
// directly, so the gate is independent of which frontend produces a body.

TEST_CASE("migrate expr round-trips for code_block", "[migrate][b2-vtrack]")
{
  use_test_ns();
  expr2tc lhs = symbol2tc(get_int_type(32), "x");
  expr2tc rhs = constant_int2tc(get_int_type(32), BigInt(1));
  std::vector<expr2tc> stmts{
    code_assign2tc(lhs, rhs),
    code_assign2tc(lhs, constant_int2tc(get_int_type(32), BigInt(2)))};
  require_expr_roundtrip(code_block2tc(stmts));
}

TEST_CASE(
  "migrate expr round-trips for code_block (empty)",
  "[migrate][b2-vtrack]")
{
  // The zero-operand corner is the one a freshly-emitted empty body would hit;
  // pinning it explicitly is cheap insurance against an iterator-on-empty bug.
  require_expr_roundtrip(code_block2tc(std::vector<expr2tc>{}));
}

TEST_CASE("migrate expr round-trips for code_cpp_catch", "[migrate][b2-vtrack]")
{
  std::vector<irep_idt> exceptions{"std::exception", "std::runtime_error"};
  require_expr_roundtrip(code_cpp_catch2tc(exceptions));
}

TEST_CASE(
  "migrate expr round-trips for pointer_capability",
  "[migrate][b2-vtrack]")
{
  // pointer_capability is the only V1 kind that had no legacy form at all
  // before this PR: it is constructed solver-side via pointer_capability2tc.
  // V1 picks the legacy id "pointer_capability" symmetrically with
  // pointer_object's existing pair, so back-then-forward round-trips.
  use_test_ns();
  expr2tc base = symbol2tc(get_int_type(32), "x");
  type2tc cap_t = unsignedbv_type2tc(64);
  require_expr_roundtrip(pointer_capability2tc(cap_t, base));
}

// Phase 4.2 construction helpers (util/migrate.h): symbol_expr2tc and
// side_effect_function_call2tc. The contract is that each is a faithful
// drop-in for the legacy constructor it replaces -- it produces the same IREP2
// node migrate_expr would yield from the legacy form -- and round-trips
// losslessly. No frontend call site is wired to them yet (Phase 4.3/4.4).

TEST_CASE(
  "symbol_expr2tc is a drop-in for the migrated legacy symbol_expr",
  "[migrate][phase4.2]")
{
  use_test_ns();
  const symbolt *sym = test_context().find_symbol("x");
  REQUIRE(sym != nullptr);

  const expr2tc via_helper = symbol_expr2tc(*sym);
  REQUIRE(is_symbol2t(via_helper));
  REQUIRE(via_helper->type == migrate_symbol_type(*sym));

  // Equal to migrating the legacy symbol_expr(sym): the helper is what the
  // ~844 symbol_expr(symbolt) sites migrate to.
  expr2tc via_legacy;
  migrate_expr(symbol_expr(*sym), via_legacy);
  REQUIRE(via_helper == via_legacy);

  require_expr_roundtrip(via_helper);
}

TEST_CASE(
  "side_effect_function_call2tc builds a function_call side-effect",
  "[migrate][phase4.2]")
{
  use_test_ns();
  const type2tc ret = get_int_type(32);
  const symbolt *xsym = test_context().find_symbol("x");
  const expr2tc fn = symbol_expr2tc(*xsym); // callee reference
  const std::vector<expr2tc> args{
    constant_int2tc(get_int_type(32), BigInt(7)),
    constant_int2tc(get_int_type(32), BigInt(9))};

  // Canonical IREP2 form: what migrate_expr yields from a real legacy
  // side_effect_expr_function_callt. The helper must reproduce it exactly.
  side_effect_expr_function_callt legacy(migrate_type_back(ret));
  legacy.function() = symbol_expr(*xsym);
  legacy.arguments().push_back(migrate_expr_back(args[0]));
  legacy.arguments().push_back(migrate_expr_back(args[1]));
  expr2tc via_legacy;
  migrate_expr(legacy, via_legacy);

  const expr2tc via_helper = side_effect_function_call2tc(ret, fn, args);
  REQUIRE(is_sideeffect2t(via_helper));

  const sideeffect2t &se = to_sideeffect2t(via_helper);
  REQUIRE(se.kind == sideeffect2t::allockind::function_call);
  REQUIRE(se.operand == fn);
  REQUIRE(se.arguments == args);
  REQUIRE(via_helper->type == ret);

  REQUIRE(via_helper == via_legacy); // faithful drop-in
  require_expr_roundtrip(via_helper);
}
