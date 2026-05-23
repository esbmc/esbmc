/*******************************************************************\

Module: Unit tests for symbolt -- specifically the legacy / IREP2
        discriminator parity that the symbol-table source-of-truth
        flip (esbmc/esbmc#4715, B2 S5a) must preserve.

\*******************************************************************/

// Tells Catch to provide a main(). Only do this once per test executable.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/migrate.h>
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
// The symbol-table source-of-truth flip (docs/irep2-symbol-table-phase5-plan.md)
// turns `symbolt::type` into IREP2 storage and derives the legacy `typet`
// lazily. Pipeline code discriminates function symbols from data symbols by
// querying `is_code` on the type: today via `sym.get_type().is_code()` (legacy
// irept walk), tomorrow via `is_code_type(sym.get_type2())` (IREP2 type-id
// check). These tests pin the property that *both queries always agree*, on
// every write path that S5a will reshape.
//
// The hazard documented in docs/irep2-symbol-table-migration-plan.md is the
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
