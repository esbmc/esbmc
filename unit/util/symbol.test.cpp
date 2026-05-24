/*******************************************************************\

Module: Unit tests for symbolt -- specifically the legacy / IREP2
        discriminator parity that the symbol-table source-of-truth
        flip (esbmc/esbmc#4715, B2 S5a) must preserve.

\*******************************************************************/

// Tells Catch to provide a main(). Only do this once per test executable.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/migrate.h>
#include <util/std_code.h>
#include <util/symbol.h>

SCENARIO(
  "Constructed symbol validity checks",
  "[core][utils][symbol__validity_checks]")
{
  GIVEN("A valid symbol")
  {
    symbolt symbol;
    irep_idt symbol_name = "Test_TestBase";
    symbol.name = symbol_name;
    // symbol.base_name = "TestBase";
    symbol.module = "TestModule";
    symbol.mode = "C";

    THEN("Symbol should be well formed")
    {
      //REQUIRE(symbol.is_well_formed());
    }
  }
}

// ---------------------------------------------------------------------------
// is_code discriminator parity (B2 S5a precondition).
//
// The symbol-table source-of-truth flip (docs/irep2-migration.md, B2 §S5a)
// turns `symbolt::type` into IREP2 storage and derives the legacy `typet`
// lazily. Pipeline code discriminates function symbols from data symbols by
// querying `is_code` on the type: today via `sym.get_type().is_code()` (legacy
// irept walk), tomorrow via `is_code_type(sym.get_type2())` (IREP2 type-id
// check). These tests pin the property that *both queries always agree*, on
// every write path that S5a will reshape.
//
// The hazard documented in docs/irep2-migration.md is the
// dual-role / static-lifetime case: a function symbol and a same-named
// static_lifetime global have different `is_code` answers despite sharing
// `base_name`. Both forms of the discriminator must distinguish them.
// ---------------------------------------------------------------------------

namespace
{
// A minimal `int f(void)` code type, the shape goto-convert sees on every
// function symbol.
typet make_legacy_code_type()
{
  code_typet ct;
  ct.return_type() = int_type();
  return ct;
}

// A signed 32-bit integer type, the shape of an ordinary data symbol.
typet make_legacy_int_type()
{
  return int_type();
}
} // namespace

TEST_CASE(
  "symbolt: legacy is_code matches IREP2 is_code_type for a function symbol",
  "[core][utils][symbol][b2-s5a]")
{
  symbolt sym;
  sym.id = sym.name = "fn";
  sym.mode = "C";
  sym.set_type(make_legacy_code_type());

  // Legacy path: irept walk on the cached/stored typet.
  REQUIRE(sym.get_type().is_code());

  // IREP2 path: type-id check on the cached/stored type2tc. Going through the
  // chokepoint also exercises the NDEBUG round-trip cross-check.
  type2tc t2 = migrate_symbol_type(sym);
  REQUIRE(is_code_type(t2));

  // Direct accessor (S4b cache) must agree with the chokepoint reading.
  REQUIRE(sym.get_type2() == t2);
}

TEST_CASE(
  "symbolt: legacy is_code is false for a non-code data symbol, IREP2 agrees",
  "[core][utils][symbol][b2-s5a]")
{
  symbolt sym;
  sym.id = sym.name = "g";
  sym.mode = "C";
  sym.static_lifetime = true;
  sym.set_type(make_legacy_int_type());

  REQUIRE_FALSE(sym.get_type().is_code());
  REQUIRE_FALSE(is_code_type(migrate_symbol_type(sym)));
  REQUIRE_FALSE(is_code_type(sym.get_type2()));
}

TEST_CASE(
  "symbolt: dual-role -- function + same-named static_lifetime global stay "
  "distinct under both discriminators",
  "[core][utils][symbol][b2-s5a]")
{
  // Same `base_name` (the colliding bit). Distinct fully-qualified ids -- the
  // symbol table keys on id, but is_code is what pipeline lookups branch on.
  symbolt fn;
  fn.id = "c:test.c@F@f";
  fn.name = "f";
  fn.mode = "C";
  fn.set_type(make_legacy_code_type());

  symbolt g;
  g.id = "c:@f";
  g.name = "f";
  g.mode = "C";
  g.static_lifetime = true;
  g.set_type(make_legacy_int_type());

  // Both discriminators must agree per-symbol.
  REQUIRE(fn.get_type().is_code());
  REQUIRE(is_code_type(fn.get_type2()));
  REQUIRE_FALSE(g.get_type().is_code());
  REQUIRE_FALSE(is_code_type(g.get_type2()));

  // And they must not collapse: distinct symbols => distinct IREP2 type forms.
  REQUIRE_FALSE(fn.get_type2() == g.get_type2());
}

TEST_CASE(
  "symbolt: legacy set_type invalidates the IREP2 cache; rewriting "
  "function->int->function flips is_code consistently on both sides",
  "[core][utils][symbol][b2-s5a]")
{
  symbolt sym;
  sym.id = sym.name = "x";
  sym.mode = "C";
  sym.set_type(make_legacy_code_type());
  REQUIRE(sym.get_type().is_code());
  REQUIRE(is_code_type(sym.get_type2()));

  // Overwrite via the legacy setter: cache must invalidate so the next IREP2
  // read reflects the new type.
  sym.set_type(make_legacy_int_type());
  REQUIRE_FALSE(sym.get_type().is_code());
  REQUIRE_FALSE(is_code_type(sym.get_type2()));

  // And back.
  sym.set_type(make_legacy_code_type());
  REQUIRE(sym.get_type().is_code());
  REQUIRE(is_code_type(sym.get_type2()));
}

TEST_CASE(
  "symbolt: IREP2-side set_type stores the cache and derives a legacy typet "
  "whose is_code agrees",
  "[core][utils][symbol][b2-s5a]")
{
  symbolt sym;
  sym.id = sym.name = "y";
  sym.mode = "C";

  // Build an IREP2 code type directly and store it via the IREP2 setter.
  // set_symbol_type (util/migrate.h) routes through symbolt::set_type(type2tc),
  // which stores the cache authoritatively and derives the legacy field via
  // migrate_type_back exactly once.
  type2tc fn_t = code_type2tc(
    std::vector<type2tc>{},
    int_type2(),
    std::vector<irep_idt>{},
    /*ellipsis=*/false);
  set_symbol_type(sym, fn_t);
  REQUIRE(is_code_type(sym.get_type2()));
  REQUIRE(sym.get_type().is_code());

  // Replace with a non-code IREP2 type; same discriminator agreement.
  set_symbol_type(sym, int_type2());
  REQUIRE_FALSE(is_code_type(sym.get_type2()));
  REQUIRE_FALSE(sym.get_type().is_code());
}

TEST_CASE(
  "symbolt: swap exchanges both the legacy and IREP2 cache, preserving "
  "is_code on each side of the swap",
  "[core][utils][symbol][b2-s5a]")
{
  symbolt a;
  a.id = a.name = "a";
  a.mode = "C";
  a.set_type(make_legacy_code_type());

  symbolt b;
  b.id = b.name = "b";
  b.mode = "C";
  b.set_type(make_legacy_int_type());

  // Pre-swap: warm the IREP2 cache so we are also testing cache-swap, not
  // just legacy-swap-then-rederive.
  REQUIRE(is_code_type(a.get_type2()));
  REQUIRE_FALSE(is_code_type(b.get_type2()));

  a.swap(b);

  // Identifiers swap (sanity), and so do the discriminators on both sides.
  REQUIRE(a.id == "b");
  REQUIRE(b.id == "a");
  REQUIRE_FALSE(a.get_type().is_code());
  REQUIRE_FALSE(is_code_type(a.get_type2()));
  REQUIRE(b.get_type().is_code());
  REQUIRE(is_code_type(b.get_type2()));
}

// ---------------------------------------------------------------------------
// Pre-V2 value-side parity (B2 V-track, docs/irep2-migration.md §V-track).
//
// V2 will mirror S5a on the value side: `expr2tc value_` becomes the dominant
// representation, the legacy `exprt` is derived lazily via migrate_expr_back
// (now safe for code_block thanks to V1, #4737). The tests below pin the
// invariant V2 must preserve: the legacy `get_value().is_code()` query and an
// equivalent IREP2-side query always agree, on every write path V2 will
// reshape. They are written so they pass *today* under S4b's value layout and
// will continue to pass after V2 -- anything in between is a real bug.
//
// Mirror of the type-side dual-role tests above (#4733). The "is body"
// discriminator is the value-side analogue of "is code-typed": for a function
// symbol the legacy value is a `code_blockt` (id "code") and the IREP2 value
// is a `code_block2t`; for a data symbol the legacy value is an initialiser
// expr (or nil) and the IREP2 value is the corresponding kind. We exercise
// both discriminators side-by-side.
// ---------------------------------------------------------------------------

namespace
{
// Build a legacy code_blockt with two trivial assign statements. The shape a
// function body has after goto-convert.
codet make_legacy_body()
{
  code_blockt block;
  // The block needs at least one statement so the round-trip through migrate
  // exercises a non-empty operands vector (V1 round-trip covers this case in
  // unit/util/migrate.test.cpp).
  codet skip("skip");
  block.move_to_operands(skip);
  return block;
}

// Test whether a possibly-nil expr2tc is a code_block2t. The generated
// is_code_block2t() helper derefs t->expr_id, so it null-derefs on a nil
// container -- guard nil explicitly.
bool is_block(const expr2tc &e)
{
  return !is_nil_expr(e) && is_code_block2t(e);
}
} // namespace

TEST_CASE(
  "symbolt: legacy is_code on a body value matches IREP2 is_code_block2t",
  "[core][utils][symbol][b2-vtrack]")
{
  symbolt sym;
  sym.id = sym.name = "fn";
  sym.mode = "C";
  sym.set_type(make_legacy_code_type());
  sym.set_value(make_legacy_body());

  // Legacy: value.id() == "code".
  REQUIRE(sym.get_value().is_code());

  // IREP2: code_block2t. Going through the IREP2 accessor warms the cache
  // (S4b shape) or returns the stored field (post-V2 shape) -- same answer
  // either way.
  REQUIRE(is_block(sym.get_value2()));
}

TEST_CASE(
  "symbolt: legacy is_code on a constant initialiser is false, IREP2 agrees",
  "[core][utils][symbol][b2-vtrack]")
{
  symbolt sym;
  sym.id = sym.name = "g";
  sym.mode = "C";
  sym.static_lifetime = true;
  sym.set_type(make_legacy_int_type());
  sym.set_value(from_integer(42, int_type()));

  REQUIRE_FALSE(sym.get_value().is_code());
  REQUIRE_FALSE(is_block(sym.get_value2()));

  // The IREP2 form should be a constant_int2t -- the symmetric positive
  // check, not just "not a block".
  REQUIRE(is_constant_int2t(sym.get_value2()));
}

TEST_CASE(
  "symbolt: default-constructed symbol has nil value on both sides",
  "[core][utils][symbol][b2-vtrack]")
{
  symbolt sym;
  REQUIRE(sym.get_value().is_nil());
  REQUIRE(is_nil_expr(sym.get_value2()));
}

TEST_CASE(
  "symbolt: dual-role -- function with body + same-named static_lifetime "
  "global with initialiser stay distinct under both value discriminators",
  "[core][utils][symbol][b2-vtrack]")
{
  // Same `base_name` (the colliding bit). Distinct fully-qualified ids -- the
  // symbol table keys on id, but is_code on the value is what body-walkers
  // such as goto_convert_functions branch on.
  symbolt fn;
  fn.id = "c:test.c@F@f";
  fn.name = "f";
  fn.mode = "C";
  fn.set_type(make_legacy_code_type());
  fn.set_value(make_legacy_body());

  symbolt g;
  g.id = "c:@f";
  g.name = "f";
  g.mode = "C";
  g.static_lifetime = true;
  g.set_type(make_legacy_int_type());
  g.set_value(from_integer(0, int_type()));

  // Per-symbol parity.
  REQUIRE(fn.get_value().is_code());
  REQUIRE(is_block(fn.get_value2()));
  REQUIRE_FALSE(g.get_value().is_code());
  REQUIRE_FALSE(is_block(g.get_value2()));

  // And the IREP2 forms must not collapse across symbols.
  REQUIRE_FALSE(fn.get_value2() == g.get_value2());
}

TEST_CASE(
  "symbolt: legacy set_value invalidates the IREP2 value cache; flipping "
  "body->int->body keeps is_code parity on both sides",
  "[core][utils][symbol][b2-vtrack]")
{
  symbolt sym;
  sym.id = sym.name = "x";
  sym.mode = "C";
  sym.set_type(make_legacy_code_type());

  sym.set_value(make_legacy_body());
  REQUIRE(sym.get_value().is_code());
  REQUIRE(is_block(sym.get_value2()));

  // Overwrite with a non-code value: the IREP2 cache must invalidate.
  sym.set_value(from_integer(1, int_type()));
  REQUIRE_FALSE(sym.get_value().is_code());
  REQUIRE_FALSE(is_block(sym.get_value2()));

  // And back.
  sym.set_value(make_legacy_body());
  REQUIRE(sym.get_value().is_code());
  REQUIRE(is_block(sym.get_value2()));
}

TEST_CASE(
  "symbolt: swap exchanges the legacy value and the IREP2 value cache, "
  "preserving is_code on each side of the swap",
  "[core][utils][symbol][b2-vtrack]")
{
  symbolt a;
  a.id = a.name = "a";
  a.mode = "C";
  a.set_type(make_legacy_code_type());
  a.set_value(make_legacy_body());

  symbolt b;
  b.id = b.name = "b";
  b.mode = "C";
  b.set_type(make_legacy_int_type());
  b.set_value(from_integer(7, int_type()));

  // Pre-swap: warm the IREP2 value cache on both sides so swap is exercised
  // with both caches populated -- not just legacy-swap-then-rederive.
  REQUIRE(is_block(a.get_value2()));
  REQUIRE_FALSE(is_block(b.get_value2()));

  a.swap(b);

  REQUIRE(a.id == "b");
  REQUIRE(b.id == "a");
  REQUIRE_FALSE(a.get_value().is_code());
  REQUIRE_FALSE(is_block(a.get_value2()));
  REQUIRE(b.get_value().is_code());
  REQUIRE(is_block(b.get_value2()));
}
