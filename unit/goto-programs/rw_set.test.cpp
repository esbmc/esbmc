// Targeted unit tests for the IREP2-native rw_set data-race analysis
// (esbmc/esbmc#4715, B5 Phase 2.1/2.2 -- #4718/#4719). Exercise
// `rw_sett::read_rec` / `compute` on hand-built `expr2tc` trees and
// assert the `entries` map content directly, with no frontend in the
// loop. Pin the paths the migration reshaped:
//   - plain symbol read/write registration
//   - is_index2t      : array access, suffix unchanged, object = base name
//   - is_member2t     : struct member, suffix = "." + member name
//   - is_if2t         : guard split (both branches recorded with their
//                       respective true/false guards)
//   - is_address_of2t : taking an address does NOT register an access
//   - is_code_assign2t: rhs read + lhs write via compute()
// Companion regression in regression/esbmc/github_4715_rwset[_fail]/
// covers the end-to-end behaviour via --data-races-check on the
// index/member/if paths.

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <goto-programs/rw_set.h>
#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/c_types.h>
#include <util/context.h>
#include <util/migrate.h>
#include <util/namespace.h>

namespace
{
// Build a contextt with the symbols the tests reference, all marked
// static_lifetime so rw_set's filter at read_write_rec accepts them as
// race-eligible (the legacy filter only registers shared state).
contextt &rwset_test_context()
{
  static contextt ctx;
  static bool initialised = false;
  if (!initialised)
  {
    auto add_global = [](contextt &c, const std::string &n, const type2tc &t) {
      symbolt s;
      s.id = s.name = n;
      s.mode = "C";
      s.static_lifetime = true;
      s.lvalue = true;
      set_symbol_type(s, t);
      c.add(s);
    };

    type2tc int32 = get_int_type(32);
    type2tc arr_t = array_type2tc(
      int32, constant_int2tc(get_uint_type(64), BigInt(4)), /*inf=*/false);
    type2tc struct_t = struct_type2tc(
      std::vector<type2tc>{int32, int32},
      std::vector<irep_idt>{"x", "y"},
      std::vector<irep_idt>{"x", "y"},
      "s");

    add_global(ctx, "x", int32);
    add_global(ctx, "y", int32);
    add_global(ctx, "arr", arr_t);
    add_global(ctx, "g_struct", struct_t);
    add_global(ctx, "cond", get_bool_type());

    initialised = true;
  }
  return ctx;
}

// Point the migrate layer at our namespace (rw_set lookups go through it).
const namespacet &rwset_test_ns()
{
  static namespacet ns(rwset_test_context());
  migrate_namespace_lookup = &ns;
  return ns;
}

// Build a minimal goto_programt with one ASSERT instruction so the
// rw_sett constructor has a valid `const_targett` to store. `read_rec`
// never dereferences `target`, so the instruction kind is immaterial
// for these tests -- but the iterator must be non-singular.
goto_programt &rwset_test_program()
{
  static goto_programt prog;
  static bool initialised = false;
  if (!initialised)
  {
    auto t = prog.add_instruction();
    t->type = ASSERT;
    initialised = true;
  }
  return prog;
}

// Convenience: build an rw_sett bound to the test namespace and program.
rw_sett make_rw_set()
{
  return rw_sett(rwset_test_ns(), rwset_test_program().instructions.begin());
}

// Convenience: a symbol2tc for one of the globals registered above.
expr2tc sym(const std::string &name)
{
  const symbolt *s = rwset_test_context().find_symbol(name);
  REQUIRE(s != nullptr);
  return symbol2tc(s->get_type2(), name);
}
} // namespace

TEST_CASE(
  "rw_set: plain symbol read registers an r entry",
  "[core][goto-programs][rw_set][b5-phase2]")
{
  rw_sett rw = make_rw_set();
  rw.read_rec(sym("x"));

  REQUIRE(rw.entries.size() == 1);
  auto it = rw.entries.find("x");
  REQUIRE(it != rw.entries.end());
  REQUIRE(it->second.object == irep_idt("x"));
  REQUIRE(it->second.r);
  REQUIRE_FALSE(it->second.w);
  REQUIRE_FALSE(it->second.deref);
}

TEST_CASE(
  "rw_set: array-index access registers the base symbol, suffix unchanged",
  "[core][goto-programs][rw_set][b5-phase2]")
{
  // arr[0]: is_index2t around symbol2tc(arr). Per rw_set.cpp the recursion
  // descends into source_value and DROPS the index from the suffix -- the
  // object key is "arr" verbatim.
  expr2tc idx_zero = constant_int2tc(get_uint_type(64), BigInt(0));
  expr2tc index_expr = index2tc(get_int_type(32), sym("arr"), idx_zero);

  rw_sett rw = make_rw_set();
  rw.read_rec(index_expr);

  REQUIRE(rw.entries.size() == 1);
  auto it = rw.entries.find("arr");
  REQUIRE(it != rw.entries.end());
  REQUIRE(it->second.r);
  REQUIRE_FALSE(it->second.w);
  // original_expr is preserved as the index expression so downstream
  // race-check instrumentation knows the access shape.
  REQUIRE(it->second.original_expr == index_expr);
}

TEST_CASE(
  "rw_set: struct member access registers the symbol with .field suffix",
  "[core][goto-programs][rw_set][b5-phase2]")
{
  // g_struct.x: is_member2t around symbol2tc(g_struct). The recursion
  // appends "." + member_name to the suffix as it descends.
  expr2tc member_expr =
    member2tc(get_int_type(32), sym("g_struct"), irep_idt("x"));

  rw_sett rw = make_rw_set();
  rw.read_rec(member_expr);

  REQUIRE(rw.entries.size() == 1);
  // Key is "g_struct.x" (symbol id + ".x" suffix).
  auto it = rw.entries.find("g_struct.x");
  REQUIRE(it != rw.entries.end());
  REQUIRE(it->second.r);
  REQUIRE_FALSE(it->second.w);
}

TEST_CASE(
  "rw_set: if-expression splits the guard on the two branches",
  "[core][goto-programs][rw_set][b5-phase2]")
{
  // if (cond) x else y -- read the result. Per rw_set.cpp the cond is
  // read unconditionally; the branches each get a guard with the cond
  // and its negation respectively.
  expr2tc if_expr = if2tc(get_int_type(32), sym("cond"), sym("x"), sym("y"));

  rw_sett rw = make_rw_set();
  rw.read_rec(if_expr);

  // Three entries: cond (the discriminator) plus x and y on the two branches.
  REQUIRE(rw.entries.size() == 3);
  REQUIRE(rw.entries.find("cond") != rw.entries.end());
  REQUIRE(rw.entries.find("x") != rw.entries.end());
  REQUIRE(rw.entries.find("y") != rw.entries.end());

  // The cond entry's guard is the empty / outer guard (no parent if-split
  // is in scope on the top-level call).
  // The x branch carries a guard with cond; the y branch carries a guard
  // with not(cond). Both guards must be non-trivial (their as_expr() form
  // is not the bare cond, because the guard is the accumulated chain).
  REQUIRE(rw.entries["x"].guard != rw.entries["y"].guard);
}

TEST_CASE(
  "rw_set: address-of does NOT register a read",
  "[core][goto-programs][rw_set][b5-phase2]")
{
  // &x in expression position is not a read of x; rw_set.cpp explicitly
  // short-circuits is_address_of2t.
  expr2tc addr = address_of2tc(pointer_type2tc(get_int_type(32)), sym("x"));

  rw_sett rw = make_rw_set();
  rw.read_rec(addr);

  REQUIRE(rw.entries.empty());
}

TEST_CASE(
  "rw_set: code_assign separates rhs (read) from lhs (write)",
  "[core][goto-programs][rw_set][b5-phase2]")
{
  // x = y -- via the public assign() helper that compute() would call
  // for a code_assign2t. Asserts the rhs registers as r-only and the
  // lhs as w-only.
  rw_sett rw = make_rw_set();
  // Use the protected path via rw_set::compute on a code_assign2tc.
  // compute() reads instruction.is_goto()/is_assert()/is_assume() only on
  // the read_rec fallback arm; for code_assign2t it dispatches directly.
  expr2tc assign = code_assign2tc(sym("x"), sym("y"));
  rw.compute(assign);

  REQUIRE(rw.entries.size() == 2);
  REQUIRE(rw.entries["y"].r);
  REQUIRE_FALSE(rw.entries["y"].w);
  REQUIRE(rw.entries["x"].w);
  REQUIRE_FALSE(rw.entries["x"].r);
}
