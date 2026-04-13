#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../testing-utils/goto_factory.h"
#include <goto-programs/goto_atomicity_check.h>

/// Returns true iff @p id names a user-defined function (not an ESBMC runtime
/// or C library function).  User functions have IDs of the form c:@F@name
/// where name does not begin with __ or come from a library source file.
static bool is_user_function(const irep_idt &id)
{
  const std::string &s = id.as_string();
  return s.rfind("c:@F@", 0) == 0 &&
         s.find("__ESBMC") == std::string::npos &&
         s.find("__VERIFIER") == std::string::npos;
}

/// Counts ASSERT instructions whose property tag is "atomicity"
/// in user-defined functions only (excludes runtime/library functions).
static unsigned count_atomicity_asserts(const goto_functionst &functions)
{
  unsigned n = 0;
  for (const auto &kv : functions.function_map)
  {
    if (!is_user_function(kv.first))
      continue;
    for (const auto &instr : kv.second.body.instructions)
      if (instr.is_assert() && instr.location.property() == "atomicity")
        n++;
  }
  return n;
}

/// Returns true iff any user-defined function body contains an ATOMIC_BEGIN.
static bool has_atomic_begin(const goto_functionst &functions)
{
  for (const auto &kv : functions.function_map)
  {
    if (!is_user_function(kv.first))
      continue;
    for (const auto &instr : kv.second.body.instructions)
      if (instr.is_atomic_begin())
        return true;
  }
  return false;
}

/// Parses @p code, runs goto_atomicity_check, and returns the
/// resulting program (with instrumentation applied).
static program run_atomicity_check(std::string code)
{
  program P =
    goto_factory::get_goto_functions(code, goto_factory::Architecture::BIT_32);
  goto_atomicity_check(P.functions, P.ns, P.context);
  return P;
}

// ---------------------------------------------------------------------------
// is_global / count_globals / collect_globals — via instrument_assign
// ---------------------------------------------------------------------------

SCENARIO(
  "no globals: no atomicity assertions are inserted",
  "[atomicity_check]")
{
  // All variables are locals → count_globals == 0 for every ASSIGN.
  // instrument_assign returns immediately after collect_globals finds nothing.
  std::string code = R"(
    int main() {
      int a = 1;
      int b = a + 2;
      return b;
    }
  )";

  auto P = run_atomicity_check(code);
  REQUIRE(count_atomicity_asserts(P.functions) == 0);
  REQUIRE_FALSE(has_atomic_begin(P.functions));
}

SCENARIO(
  "global on RHS, local LHS: assertion inserted without ATOMIC_BEGIN",
  "[atomicity_check]")
{
  // instrument_assign: lhs_globals == 0 (local x), source has global g.
  // need_atomic = false → no ATOMIC_BEGIN/END.
  // Covers: collect_globals symbol2t path (direct global read).
  std::string code = R"(
    int g = 0;
    int main() {
      int x = g;
      return x;
    }
  )";

  auto P = run_atomicity_check(code);
  REQUIRE(count_atomicity_asserts(P.functions) >= 1);
  REQUIRE_FALSE(has_atomic_begin(P.functions));
}

SCENARIO(
  "global on both LHS and RHS: assertion inserted with ATOMIC_BEGIN/END",
  "[atomicity_check]")
{
  // instrument_assign: lhs_globals > 0 (global g), source has global g.
  // need_atomic = true → ATOMIC_BEGIN inserted before assertion,
  // ATOMIC_END inserted after the original ASSIGN.
  // Covers: need_atomic = true branch.
  std::string code = R"(
    int g = 0;
    int main() {
      g = g + 1;
      return 0;
    }
  )";

  auto P = run_atomicity_check(code);
  REQUIRE(count_atomicity_asserts(P.functions) >= 1);
  REQUIRE(has_atomic_begin(P.functions));
}

SCENARIO(
  "address-of a global on RHS: no atomicity assertion",
  "[atomicity_check]")
{
  // collect_globals skips address_of2t sub-expressions.
  // Taking the address of a global is not a data race.
  // Covers: is_address_of2t early-return branch.
  std::string code = R"(
    int g = 0;
    int main() {
      int *p = &g;
      return 0;
    }
  )";

  auto P = run_atomicity_check(code);
  REQUIRE(count_atomicity_asserts(P.functions) == 0);
  REQUIRE_FALSE(has_atomic_begin(P.functions));
}

SCENARIO(
  "dereference of a global pointer on RHS: assertion inserted",
  "[atomicity_check]")
{
  // collect_globals dereference2t path: checks that the pointer operand p
  // is a global symbol, then snapshots *p as the expression to protect.
  // Covers: is_dereference2t branch in collect_globals.
  std::string code = R"(
    int g;
    int *p;
    int main() {
      p = &g;
      int x = *p;
      return x;
    }
  )";

  auto P = run_atomicity_check(code);
  REQUIRE(count_atomicity_asserts(P.functions) >= 1);
}

SCENARIO(
  "member access of a global struct on RHS: assertion inserted",
  "[atomicity_check]")
{
  // collect_globals member2t path: checks that source_value of the member
  // expression is a global symbol, then snapshots s.a.
  // Covers: is_member2t branch in collect_globals.
  std::string code = R"(
    struct S { int a; int b; };
    struct S s;
    int main() {
      int x = s.a;
      return x;
    }
  )";

  auto P = run_atomicity_check(code);
  REQUIRE(count_atomicity_asserts(P.functions) >= 1);
}

SCENARIO(
  "array index is a global: assertion inserted",
  "[atomicity_check]")
{
  // collect_globals index2t path: when the index is a global symbol,
  // the whole indexed expression is snapshotted.
  // Covers: is_index2t branch with global index.
  std::string code = R"(
    int gi;
    int arr[10];
    int main() {
      int x = arr[gi];
      return x;
    }
  )";

  auto P = run_atomicity_check(code);
  REQUIRE(count_atomicity_asserts(P.functions) >= 1);
}

// ---------------------------------------------------------------------------
// instrument_guard — ASSUME path
// ---------------------------------------------------------------------------

SCENARIO(
  "ASSUME with a global guard: atomicity assertion prepended",
  "[atomicity_check]")
{
  // instrument_guard ASSUME branch: guard_expr = it->guard (the assumption).
  // When g is global and the guard is simple (leaf after typecast),
  // should_instrument_guard returns true and collect_globals finds g.
  // Covers: is_assume() branch in instrument_guard.
  std::string code = R"(
    int g;
    int main() {
      __ESBMC_assume(g);
      return 0;
    }
  )";

  auto P = run_atomicity_check(code);
  REQUIRE(count_atomicity_asserts(P.functions) >= 1);
}

// ---------------------------------------------------------------------------
// instrument_guard — GOTO path
// ---------------------------------------------------------------------------

SCENARIO(
  "GOTO with a _Bool global guard: atomicity assertion prepended",
  "[atomicity_check]")
{
  // if(g) generates IF !g THEN GOTO. Guard = NOT(g). NOT's first operand
  // is g (simple leaf) → should_instrument_guard returns true.
  // Covers: is_goto() branch in instrument_guard.
  std::string code = R"(
    _Bool g;
    int main() {
      if (g)
        return 1;
      return 0;
    }
  )";

  auto P = run_atomicity_check(code);
  REQUIRE(count_atomicity_asserts(P.functions) >= 1);
}

// ---------------------------------------------------------------------------
// instrument_guard — RETURN path
// ---------------------------------------------------------------------------

SCENARIO(
  "RETURN with a global return value: atomicity assertion prepended",
  "[atomicity_check]")
{
  // instrument_guard RETURN branch: guard_expr = operand of code_return2t.
  // When g is global and simple, collect_globals finds it and inserts
  // an assertion snapshot before the RETURN instruction.
  // Covers: is_return() && is_code_return2t() branch.
  std::string code = R"(
    int g;
    int read_g() {
      return g;
    }
    int main() {
      return read_g();
    }
  )";

  auto P = run_atomicity_check(code);
  REQUIRE(count_atomicity_asserts(P.functions) >= 1);
}

// ---------------------------------------------------------------------------
// instrument_guard — ASSERT path
// ---------------------------------------------------------------------------

SCENARIO(
  "ASSERT with a global guard: atomicity assertion prepended",
  "[atomicity_check]")
{
  // instrument_guard ASSERT branch: ASSERT guard references global g.
  // The typecast's first (and only) operand is g (simple leaf) so
  // should_instrument_guard returns true.
  // Covers: is_assert() branch in instrument_guard.
  std::string code = R"(
    int g;
    int main() {
      __ESBMC_assert(g, "g must be nonzero");
      return 0;
    }
  )";

  auto P = run_atomicity_check(code);
  REQUIRE(count_atomicity_asserts(P.functions) >= 1);
}

// ---------------------------------------------------------------------------
// should_instrument_guard — complex first operand → false
// ---------------------------------------------------------------------------

SCENARIO(
  "complex guard (first operand has sub-expressions): not instrumented",
  "[atomicity_check]")
{
  // if(g > 0) generates IF !(g > 0) THEN GOTO. Guard = NOT(g > 0).
  // NOT's first operand is 'g > 0', which has sub-ops (g and 0).
  // should_instrument_guard returns false → no guard assertion.
  // There is also no global-writing ASSIGN, so 0 atomicity assertions total.
  // Covers: first_has_ops = true → should_instrument_guard returns false.
  std::string code = R"(
    int g;
    int main() {
      if (g > 0) {}
      return 0;
    }
  )";

  auto P = run_atomicity_check(code);
  // No global-modifying ASSIGNs and the complex guard is skipped.
  REQUIRE(count_atomicity_asserts(P.functions) == 0);
}

// ---------------------------------------------------------------------------
// goto_atomicity_check — empty body skipped
// ---------------------------------------------------------------------------

SCENARIO("empty function body is skipped by the checker", "[atomicity_check]")
{
  // goto_atomicity_check only calls checker.check() when body is non-empty.
  // A function with no statements has an empty body.
  // Covers: kv.second.body.empty() → skip branch in goto_atomicity_check.
  std::string code = R"(
    void noop() {}
    int main() { noop(); return 0; }
  )";
  auto P = run_atomicity_check(code);
  // noop() has an empty body — no atomicity assertions inserted there.
  REQUIRE(count_atomicity_asserts(P.functions) == 0);
}

// ---------------------------------------------------------------------------
// Non-global guard: collect_globals returns empty → no assertion
// ---------------------------------------------------------------------------

SCENARIO(
  "ASSUME with a non-global guard: no atomicity assertion",
  "[atomicity_check]")
{
  // instrument_guard: should_instrument_guard returns true (simple leaf
  // after equality), but collect_globals finds no globals — the equality
  // operands are local x and a constant.  conjuncts is empty → return.
  // We use x == 0 rather than bare x to avoid the typecast that would pull
  // in the rounding_mode global.
  // Covers: conjuncts.empty() early-return in instrument_guard.
  std::string code = R"(
    int main() {
      int x = 0;
      __ESBMC_assume(x == 0);
      return 0;
    }
  )";

  auto P = run_atomicity_check(code);
  REQUIRE(count_atomicity_asserts(P.functions) == 0);
}

// ---------------------------------------------------------------------------
// Non-symbol LHS for ASSIGN: comment uses "atomicity violation" fallback
// ---------------------------------------------------------------------------

SCENARIO(
  "dereference as ASSIGN LHS: uses generic atomicity violation comment",
  "[atomicity_check]")
{
  // When the ASSIGN target is not a simple symbol (e.g. *p),
  // instrument_assign uses the generic comment "atomicity violation"
  // rather than "atomicity violation on assignment to <name>".
  // Covers: !is_symbol2t(assign.target) branch in comment construction.
  std::string code = R"(
    int g;
    int *p;
    int main() {
      p = &g;
      *p = g;
      return 0;
    }
  )";

  // *p = g: target = *p (dereference, not a symbol), source = g (global).
  // lhs_globals > 0 (p is global) → need_atomic = true, ATOMIC_BEGIN added.
  // Generic comment is used.
  auto P = run_atomicity_check(code);
  REQUIRE(count_atomicity_asserts(P.functions) >= 1);
  REQUIRE(has_atomic_begin(P.functions));

  // Check that the generic comment is used (no named variable in comment).
  bool found_generic = false;
  for (const auto &kv : P.functions.function_map)
    for (const auto &instr : kv.second.body.instructions)
      if (
        instr.is_assert() &&
        instr.location.comment() == "atomicity violation" &&
        instr.location.property() == "atomicity")
        found_generic = true;
  REQUIRE(found_generic);
}
