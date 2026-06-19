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
#include <util/std_code.h>
#include <util/std_expr.h>
#include <util/c_types.h>
#include <util/arith_tools.h>

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

TEST_CASE("migrate expr round-trips for code_decl with init", "[migrate]")
{
  use_test_ns();
  expr2tc rhs = constant_int2tc(get_int_type(32), BigInt(42));
  // 2-op code_decl: declaration with initializer -- must round-trip so that
  // goto_convert places the DEAD instruction at end-of-scope, not early.
  require_expr_roundtrip(code_decl2tc(get_int_type(32), irep_idt("x"), rhs));
  // 1-op code_decl: no initializer -- init field is nil, round-trips as 1-op
  require_expr_roundtrip(code_decl2tc(get_int_type(32), irep_idt("x")));
}

TEST_CASE(
  "migrate coerces mismatched ternary branches to the result type",
  "[migrate][v4-cf]")
{
  // esbmc#4715: the C `assert` macro lowers to a discarded statement-level
  // ternary `cond ? 0 : __assert_fail()` whose result type is void (empty) and
  // whose branch types differ from it (int and void). if2t requires both
  // branches to share the result type, so the forward migration must coerce the
  // divergent branch -- otherwise the if2t constructor's type invariant trips.
  // Under --irep2-bodies the whole body is migrated before goto_convert lowers
  // the ternary, so this shape reaches migrate_expr directly.
  use_test_ns();
  typet voidt = migrate_type_back(get_empty_type());
  typet boolt = migrate_type_back(get_bool_type());
  typet intt = migrate_type_back(get_int_type(32));

  exprt tern("if", voidt);
  tern.copy_to_operands(
    from_integer(1, boolt), // cond
    from_integer(0, intt),  // true: int, differs from the void result type
    from_integer(0, intt)); // false: int, differs from the void result type

  expr2tc m;
  migrate_expr(tern, m);
  REQUIRE(is_if2t(m));
  // The if2t and both arms share the (void) result type id.
  const if2t &i = to_if2t(m);
  REQUIRE(i.type->type_id == i.true_value->type->type_id);
  REQUIRE(i.type->type_id == i.false_value->type->type_id);
}

TEST_CASE(
  "migrate flattens a single-decl labeled decl-block",
  "[migrate][v4-cf]")
{
  // esbmc#4715: `lbl: char *s = p;` is legacy label(decl-block(decl)). The
  // decl-block must NOT migrate to a code_block -- code_block back-migrates to
  // code("block"), whose scope boundary makes convert_block emit a premature
  // DEAD for the declared variable right after its init (killing it before its
  // uses). A decl-block introduces no scope (convert_decl_block), so a
  // single-decl labeled decl-block flattens to the bare decl: the label's body
  // round-trips as a code_decl, deferring the DEAD to the enclosing scope.
  use_test_ns();
  typet ct = migrate_type_back(get_int_type(32));

  codet decl("decl");
  decl.copy_to_operands(symbol_exprt("x", ct), from_integer(7, ct));
  codet declblock("decl-block");
  declblock.copy_to_operands(decl);
  codet label("label");
  label.set("label", "lbl");
  label.copy_to_operands(declblock);

  expr2tc m;
  migrate_expr(label, m);
  REQUIRE(is_code_label2t(m));
  // The labeled body is the bare decl, not a scope-introducing block.
  REQUIRE(is_code_decl2t(to_code_label2t(m).code));
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
  // Marker form (post-goto-convert CATCH-push/pop): catchable-type list only.
  std::vector<irep_idt> exceptions{"std::exception", "std::runtime_error"};
  require_expr_roundtrip(code_cpp_catch2tc(exceptions));
}

TEST_CASE(
  "migrate expr round-trips for code_cpp_catch with operands",
  "[migrate][b2-vtrack]")
{
  // Source-level try/catch (the --irep2-bodies form): operands[0] is the try
  // block, operands[1..N] the catch-handler blocks parallel to the id list.
  // The per-handler ids ride the legacy "exception_id" attribute across the
  // round-trip, so dropping them would surface here.
  std::vector<irep_idt> exceptions{"std::runtime_error", "std::logic_error"};
  std::vector<expr2tc> ops{
    code_block2tc(std::vector<expr2tc>{}),  // try block
    code_block2tc(std::vector<expr2tc>{}),  // handler for std::runtime_error
    code_block2tc(std::vector<expr2tc>{})}; // handler for std::logic_error
  require_expr_roundtrip(code_cpp_catch2tc(exceptions, ops));
}

TEST_CASE("migrate expr round-trips for code_cpp_throw", "[migrate][b2-vtrack]")
{
  // Back-arm produces sideeffect("cpp-throw"); forward arm re-lifts that to
  // code_cpp_throw2tc, completing the IREP2->legacy->IREP2 round-trip.
  use_test_ns();
  std::vector<irep_idt> exceptions{"std::runtime_error"};
  expr2tc operand = symbol2tc(get_int_type(32), "ex");
  require_expr_roundtrip(code_cpp_throw2tc(operand, exceptions));
  // rethrow: null operand, empty exception list
  require_expr_roundtrip(code_cpp_throw2tc(expr2tc(), std::vector<irep_idt>{}));
}

TEST_CASE("migrate expr round-trips for new_object", "[migrate][b2-vtrack]")
{
  // new_object is the C++ "this" placeholder in temporary_object initializers.
  // Back-migrates to exprt("new_object") carrying the type, then re-migrates
  // to the same new_object2tc.
  use_test_ns();
  type2tc t = make_struct_type();
  require_expr_roundtrip(new_object2tc(t));
}

TEST_CASE(
  "migrate expr round-trips for sideeffect temporary_object (1-op form)",
  "[migrate][b2-vtrack]")
{
  // 1-op form: operand carries the constructor call; arguments empty.
  // Back-migrates to side_effect_exprt("temporary_object") with one operand.
  use_test_ns();
  type2tc st = make_struct_type();
  expr2tc operand = symbol2tc(get_int_type(32), "x");
  expr2tc nil_expr;
  std::vector<expr2tc> no_args;
  // alloctype is the constructed struct type (as set by the C++ frontend).
  require_expr_roundtrip(sideeffect2tc(
    st,
    operand,
    nil_expr,
    no_args,
    st,
    sideeffect_allockind::temporary_object));
}

TEST_CASE(
  "migrate expr round-trips for sideeffect temporary_object (initializer form)",
  "[migrate][b2-vtrack]")
{
  // initializer-form: operand is nil; initializer stored in arguments[0].
  // Back-migrates to side_effect_exprt("temporary_object") with no direct
  // operands and the initializer restored via theexpr.initializer().
  use_test_ns();
  type2tc st = make_struct_type();
  expr2tc nil_expr;
  expr2tc init = symbol2tc(get_int_type(32), "x");
  std::vector<expr2tc> args{init};
  // alloctype is the constructed struct type (as set by the C++ frontend).
  require_expr_roundtrip(sideeffect2tc(
    st, nil_expr, nil_expr, args, st, sideeffect_allockind::temporary_object));
}

TEST_CASE(
  "migrate cpp_new[] recovers size and initializer from frontend fields",
  "[migrate][v4-cf]")
{
  // Regression for esbmc#4715 V.4.4 cpp_new parity. The C++ frontend builds
  // side_effect_exprt("cpp_new[]") with the array size in the "size" field
  // (size_irep) and the initializer in the "initializer" sub; the "#size"
  // (cmt_size) mirror is only populated later in the pipeline. Under
  // --irep2-bodies the body is migrated before that mirror runs, so forward
  // migration must read "size" — otherwise the whole size operand (and the
  // initializer) is silently dropped.
  use_test_ns();
  typet int_t = migrate_type_back(get_int_type(32));
  side_effect_exprt e("cpp_new[]", gen_pointer_type(int_t));
  e.size(symbol_exprt("x", int_t));      // array-count leaf, in "size" only
  e.initializer(from_integer(7, int_t)); // initializer in "initializer" sub

  expr2tc m;
  migrate_expr(e, m);
  REQUIRE(is_sideeffect2t(m));
  const sideeffect2t &se = to_sideeffect2t(m);
  REQUIRE(se.kind == sideeffect_allockind::cpp_new_arr);
  REQUIRE(!is_nil_expr(se.size));
  REQUIRE(se.size == symbol2tc(get_int_type(32), "x"));
  REQUIRE(se.arguments.size() == 1);
  REQUIRE(se.arguments[0] == constant_int2tc(get_int_type(32), BigInt(7)));

  // And the round-trip back must restore both the "size" field and the
  // initializer so the legacy conversion pipeline sees them.
  exprt back = migrate_expr_back(m);
  REQUIRE(back.size_irep().is_not_nil());
  REQUIRE(back.initializer().is_not_nil());
}

TEST_CASE(
  "migrate preserves source location on basic code kinds",
  "[migrate][v4-cf]")
{
  // V.4.4 (esbmc#4715): the basic code kinds (assign/decl/expression/...) carry
  // a non-reflected `location` so the source location survives the body
  // round-trip under --irep2-bodies. Without it, goto_convert stamps empty
  // instruction locations and filename-gated checks (e.g. Solidity narrowing
  // overflow, which only fires on ".sol" files) silently vanish -> unsound.
  // Equality ignores `location`, so this asserts on it explicitly rather than
  // via require_expr_roundtrip.
  use_test_ns();
  locationt loc;
  loc.set_file("contract.sol");
  loc.set_line(8);

  // code_assign: legacy codet("assign") -> code_assign2t -> codet, loc intact.
  codet assign("assign");
  assign.copy_to_operands(
    symbol_exprt("x", migrate_type_back(get_int_type(32))),
    from_integer(7, migrate_type_back(get_int_type(32))));
  assign.location() = loc;
  expr2tc m;
  migrate_expr(assign, m);
  REQUIRE(is_code_assign2t(m));
  REQUIRE(to_code_assign2t(m).location.get_file() == "contract.sol");
  exprt back = migrate_expr_back(m);
  REQUIRE(back.location().get_file() == "contract.sol");
  REQUIRE(back.location().get_line() == "8");

  // sideeffect_assign (compound `x += 10`) carries location too: it lowers to
  // an ASSIGN instruction in goto_convert, whose location gates the Solidity
  // narrowing-assignment overflow check.
  side_effect_exprt comp("assign+");
  comp.copy_to_operands(
    symbol_exprt("x", migrate_type_back(get_int_type(32))),
    from_integer(10, migrate_type_back(get_int_type(32))));
  comp.location() = loc;
  expr2tc mc;
  migrate_expr(comp, mc);
  REQUIRE(is_sideeffect_assign2t(mc));
  REQUIRE(to_sideeffect_assign2t(mc).location.get_file() == "contract.sol");
  REQUIRE(migrate_expr_back(mc).location().get_file() == "contract.sol");

  // code_assert: its location becomes the ASSERT instruction location, which
  // gates --assertion-coverage instrumentation (filename-pooled) and the
  // coverage assert-count -- a dropped location yields "Total Asserts: 0".
  codet assert_code("assert");
  assert_code.copy_to_operands(
    symbol_exprt("x", migrate_type_back(get_bool_type())));
  assert_code.location() = loc;
  expr2tc ma;
  migrate_expr(assert_code, ma);
  REQUIRE(is_code_assert2t(ma));
  REQUIRE(to_code_assert2t(ma).location.get_file() == "contract.sol");
  REQUIRE(migrate_expr_back(ma).location().get_file() == "contract.sol");

  // code_assume: same, lowers to an ASSUME instruction.
  codet assume_code("assume");
  assume_code.copy_to_operands(
    symbol_exprt("x", migrate_type_back(get_bool_type())));
  assume_code.location() = loc;
  expr2tc mu;
  migrate_expr(assume_code, mu);
  REQUIRE(is_code_assume2t(mu));
  REQUIRE(to_code_assume2t(mu).location.get_file() == "contract.sol");
  REQUIRE(migrate_expr_back(mu).location().get_file() == "contract.sol");
}

TEST_CASE(
  "migrate preserves code_block end_location",
  "[migrate][v4-cf]")
{
  // W1 (esbmc#4715): code_block2t carries a non-reflected end_location (the
  // closing-brace location) so it survives the --irep2-bodies round-trip.
  // convert_block stamps it onto the destructor-unwind instructions at scope
  // exit; without it those instructions are unlocated. Equality ignores it, so
  // assert on it explicitly.
  use_test_ns();
  locationt loc, end_loc;
  loc.set_file("contract.sol");
  loc.set_line(3);
  end_loc.set_file("contract.sol");
  end_loc.set_line(9);

  codet block("code");
  block.statement("block");
  block.location() = loc;
  block.end_location(end_loc);

  expr2tc m;
  migrate_expr(block, m);
  REQUIRE(is_code_block2t(m));
  REQUIRE(to_code_block2t(m).end_location.get_line() == "9");

  exprt back = migrate_expr_back(m);
  REQUIRE(back.location().get_line() == "3");
  REQUIRE(
    static_cast<const locationt &>(back.end_location()).get_line() == "9");
}

TEST_CASE(
  "migrate round-trips sizeof(T) carrying the measured type",
  "[migrate][v4-cf]")
{
  // esbmc#5337: sizeof(T) is an unfolded `sizeof` node whose single type_exprt
  // operand carries the measured type T. migrate maps it to a first-class
  // sizeof2t with a reflected sizeof_type field (no side-channel attribute on a
  // folded constant), and back to the legacy operand-typed form. get_alloc_type
  // reads T off this node so malloc(sizeof(T)) allocates a struct T, not char.
  use_test_ns();

  exprt sz("sizeof", size_type());
  sz.copy_to_operands(type_exprt(symbol_typet("tag-s")));
  sz.copy_to_operands(from_integer(16, size_type()));
  expr2tc m;
  migrate_expr(sz, m);
  REQUIRE(is_sizeof2t(m));
  REQUIRE(!is_nil_type(to_sizeof2t(m).sizeof_type));
  REQUIRE(is_constant_int2t(to_sizeof2t(m).value));

  exprt back = migrate_expr_back(m);
  REQUIRE(back.id() == "sizeof");
  REQUIRE(back.operands().size() == 2);
  REQUIRE(back.op0().type().id() == "symbol");

  // A plain integer constant is unaffected and stays a constant_int2t.
  expr2tc plain = constant_int2tc(get_uint_type(32), BigInt(8));
  REQUIRE(is_constant_int2t(plain));
  REQUIRE(migrate_expr_back(plain).id() == "constant");
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

// V.4 of the migration (esbmc/esbmc#4715): structured control-flow code kinds.
// These are dead-but-tested infrastructure -- nothing in the pipeline builds
// them yet (the converter emits legacy structured codet, which goto_convert
// lowers). The round-trip property pinned here is the precondition for the
// goto_convert wiring phase that lets a frontend emit IREP2 bodies. The
// nil-operand variants (else-less if, head-less for) exercise the null<->nil
// mapping that an emitted body would hit.

// A small legacy-shaped IREP2 body used as a sub-statement of the CF kinds.
static expr2tc cf_block()
{
  expr2tc lhs = symbol2tc(get_int_type(32), "x");
  expr2tc rhs = constant_int2tc(get_int_type(32), BigInt(1));
  return code_block2tc(std::vector<expr2tc>{code_assign2tc(lhs, rhs)});
}

TEST_CASE("migrate expr round-trips for code_ifthenelse", "[migrate][v4-cf]")
{
  use_test_ns();
  require_expr_roundtrip(
    code_ifthenelse2tc(gen_true_expr(), cf_block(), cf_block()));
  // No else branch: the else_case is a null expr2tc <-> nil exprt.
  require_expr_roundtrip(
    code_ifthenelse2tc(gen_true_expr(), cf_block(), expr2tc()));
}

TEST_CASE("migrate expr round-trips for code_while", "[migrate][v4-cf]")
{
  use_test_ns();
  require_expr_roundtrip(code_while2tc(gen_true_expr(), cf_block()));
}

TEST_CASE("migrate expr round-trips for code_dowhile", "[migrate][v4-cf]")
{
  use_test_ns();
  require_expr_roundtrip(code_dowhile2tc(gen_true_expr(), cf_block()));
}

TEST_CASE("migrate expr round-trips for code_for", "[migrate][v4-cf]")
{
  use_test_ns();
  expr2tc i = symbol2tc(get_int_type(32), "i");
  expr2tc init =
    code_assign2tc(i, constant_int2tc(get_int_type(32), BigInt(0)));
  expr2tc iter =
    code_assign2tc(i, constant_int2tc(get_int_type(32), BigInt(1)));
  require_expr_roundtrip(code_for2tc(init, gen_true_expr(), iter, cf_block()));
  // Absent init/cond/iter -- the null<->nil corner for every head slot.
  require_expr_roundtrip(
    code_for2tc(expr2tc(), expr2tc(), expr2tc(), cf_block()));
}

TEST_CASE("migrate expr round-trips for code_switch", "[migrate][v4-cf]")
{
  use_test_ns();
  expr2tc value = symbol2tc(get_int_type(32), "x");
  require_expr_roundtrip(code_switch2tc(value, cf_block()));
}

TEST_CASE(
  "migrate expr round-trips for code_break and code_continue",
  "[migrate][v4-cf]")
{
  use_test_ns();
  require_expr_roundtrip(code_break2tc());
  require_expr_roundtrip(code_continue2tc());
}

TEST_CASE("migrate expr round-trips for code_label", "[migrate][v4-cf]")
{
  use_test_ns();
  require_expr_roundtrip(code_label2tc("L1", cf_block()));
}

// V.4.1: each CF kind carries a source location through migrate in both
// directions. The location field is intentionally NOT in the fields tuple, so
// it does not enter operator== -- require_expr_roundtrip would pass even if it
// were dropped. Assert it survives explicitly via the typed accessor.
static locationt cf_loc()
{
  locationt l;
  l.set_file("cf.c");
  l.set_line(42u);
  return l;
}

static expr2tc cf_roundtrip(const expr2tc &e2)
{
  expr2tc back;
  migrate_expr(migrate_expr_back(e2), back);
  return back;
}

TEST_CASE(
  "migrate carries source location on CF code kinds",
  "[migrate][v4-cf]")
{
  use_test_ns();
  const locationt loc = cf_loc();
  expr2tc value = symbol2tc(get_int_type(32), "x");

  REQUIRE(
    to_code_ifthenelse2t(cf_roundtrip(code_ifthenelse2tc(
                           gen_true_expr(), cf_block(), expr2tc(), loc)))
      .location == loc);
  REQUIRE(
    to_code_while2t(
      cf_roundtrip(code_while2tc(gen_true_expr(), cf_block(), loc)))
      .location == loc);
  REQUIRE(
    to_code_dowhile2t(
      cf_roundtrip(code_dowhile2tc(gen_true_expr(), cf_block(), loc)))
      .location == loc);
  REQUIRE(
    to_code_for2t(cf_roundtrip(code_for2tc(
                    expr2tc(), expr2tc(), expr2tc(), cf_block(), loc)))
      .location == loc);
  REQUIRE(
    to_code_switch2t(cf_roundtrip(code_switch2tc(value, cf_block(), loc)))
      .location == loc);
  REQUIRE(to_code_break2t(cf_roundtrip(code_break2tc(loc))).location == loc);
  REQUIRE(
    to_code_continue2t(cf_roundtrip(code_continue2tc(loc))).location == loc);
  REQUIRE(
    to_code_label2t(cf_roundtrip(code_label2tc("L1", cf_block(), loc)))
      .location == loc);
  REQUIRE(
    to_code_assert2t(cf_roundtrip(code_assert2tc(gen_true_expr(), loc)))
      .location == loc);
  REQUIRE(
    to_code_assume2t(cf_roundtrip(code_assume2tc(gen_true_expr(), loc)))
      .location == loc);
}

TEST_CASE(
  "migrate expr round-trips for code_assert and code_assume",
  "[migrate][v4-cf]")
{
  use_test_ns();
  require_expr_roundtrip(code_assert2tc(gen_true_expr()));
  require_expr_roundtrip(code_assume2tc(gen_false_expr()));
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
