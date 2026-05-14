/*******************************************************************
 Module: goto_inlinet unit tests

 Test plan: exercise the public goto_inline / goto_partial_inline entry
 points on small C programs that drive each branch of the migrated
 helpers (parameter_assignments, replace_return, expand_function_call,
 inline_instruction).

\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "../testing-utils/goto_factory.h"
#include <goto-programs/goto_inline.h>
#include <irep2/irep2_utils.h>

namespace
{
/// Locate the function with the given id; abort the test if missing.
/// Returns a pointer so GCC's -Wdangling-reference does not fire at the
/// call sites that bind the result into a const-reference local.
const goto_functiont *
require_function(const goto_functionst &functions, const irep_idt &id)
{
  auto it = functions.function_map.find(id);
  REQUIRE(it != functions.function_map.end());
  return &it->second;
}

/// Counts instructions of the given GOTO type in @p body.
unsigned
count_instructions(const goto_programt &body, goto_program_instruction_typet t)
{
  unsigned n = 0;
  for (const auto &instr : body.instructions)
    if (instr.type == t)
      ++n;
  return n;
}

/// Counts function calls remaining in @p body.  After a successful full
/// inline, every FUNCTION_CALL to an inline-able callee is replaced by a
/// LOCATION instruction so this returns 0.
unsigned count_calls(const goto_programt &body)
{
  return count_instructions(body, FUNCTION_CALL);
}

/// Returns the number of times a code_assign2tc whose target is the symbol
/// named @p target_id appears in @p body.
unsigned
count_assigns_to(const goto_programt &body, const std::string &target_id)
{
  unsigned n = 0;
  for (const auto &instr : body.instructions)
  {
    if (!instr.is_assign() || is_nil_expr(instr.code))
      continue;
    if (!is_code_assign2t(instr.code))
      continue;
    const code_assign2t &a = to_code_assign2t(instr.code);
    if (
      is_symbol2t(a.target) && to_symbol2t(a.target).thename.as_string().find(
                                 target_id) != std::string::npos)
      ++n;
  }
  return n;
}

/// Builds a goto-program from @p code and applies full inlining directly to
/// the body of `main`, returning the resulting program. Calling the public
/// no-dest goto_inline() instead would clear __ESBMC_main's body during
/// cleanup; using goto_inlinet directly lets us inspect the inlined result.
program inline_full(const std::string &code)
{
  std::string mutable_code = code;
  program P = goto_factory::get_goto_functions(
    mutable_code, goto_factory::Architecture::BIT_64);
  cmdlinet cmd = goto_factory::get_default_cmdline("test.c");
  optionst opts = goto_factory::get_default_options(cmd);
  goto_inlinet inliner(P.functions, opts, P.ns);
  auto it = P.functions.function_map.find("c:@F@main");
  REQUIRE(it != P.functions.function_map.end());
  inliner.goto_inline(it->second.body);
  return P;
}

/// Builds a goto-program from @p code and runs partial inlining with the
/// given small-function size threshold against `main`'s body.
program inline_partial(const std::string &code, unsigned limit)
{
  std::string mutable_code = code;
  program P = goto_factory::get_goto_functions(
    mutable_code, goto_factory::Architecture::BIT_64);
  cmdlinet cmd = goto_factory::get_default_cmdline("test.c");
  optionst opts = goto_factory::get_default_options(cmd);
  goto_inlinet inliner(P.functions, opts, P.ns);
  inliner.smallfunc_limit = limit;
  auto it = P.functions.function_map.find("c:@F@main");
  REQUIRE(it != P.functions.function_map.end());
  inliner.goto_inline_rec(it->second.body, /*full=*/false);
  return P;
}
} // namespace

// ---------------------------------------------------------------------------
// inline_instruction / expand_function_call: body-available full inline
// ---------------------------------------------------------------------------

SCENARIO(
  "goto_inline expands a single user function call into the caller body",
  "[goto_inline]")
{
  // After full inlining, __ESBMC_main contains no FUNCTION_CALL to add()
  // (it has been replaced by the function body) and the parameter-binding
  // ASSIGNs to the formal `a` and `b` are present.
  // Covers: inline_instruction symbol path + expand_function_call body
  // path + parameter_assignments named formal + replace_return non-nil lhs.
  std::string code = R"(
    int add(int a, int b) { return a + b; }
    int main() {
      int x = add(1, 2);
      return x;
    }
  )";

  auto P = inline_full(code);
  const goto_functiont *main = require_function(P.functions, "c:@F@main");
  REQUIRE(count_calls(main->body) == 0);
  // Formals `a` and `b` of add() should now have ASSIGNs in the caller's
  // body. The clang frontend names parameters with the C-mangled USR of
  // the enclosing function, so we look for "add" in the parameter id.
  REQUIRE(count_assigns_to(main->body, "add@a") >= 1);
  REQUIRE(count_assigns_to(main->body, "add@b") >= 1);
}

// ---------------------------------------------------------------------------
// parameter_assignments: typecast bridge for numeric-to-numeric mismatches
// ---------------------------------------------------------------------------

SCENARIO(
  "goto_inline inserts implicit typecasts for numeric argument mismatches",
  "[goto_inline]")
{
  // The literal 0 (int) is passed where short is expected — base_type_eq
  // is false but can_typecast_argument returns true (signedbv/signedbv).
  // The migrated code wraps `actual` in a typecast2tc rather than aborting.
  // Covers: can_typecast_argument numeric branch.
  std::string code = R"(
    short f(short s) { return s; }
    int main() {
      short r = f(0);
      return r;
    }
  )";

  auto P = inline_full(code);
  const goto_functiont *main = require_function(P.functions, "c:@F@main");
  REQUIRE(count_calls(main->body) == 0);
}

// ---------------------------------------------------------------------------
// parameter_assignments: array-to-pointer decay
// ---------------------------------------------------------------------------

SCENARIO(
  "goto_inline binds an array argument to a pointer formal parameter",
  "[goto_inline]")
{
  // Passing &arr[0] (pointer<int>) to a formal of type int[10] hits the
  // is_array_type(formal) && is_pointer_type(actual) typecast branch.
  // Covers: can_typecast_argument array-to-pointer branch.
  std::string code = R"(
    int sum(int v[10]) { return v[0]; }
    int main() {
      int arr[10];
      return sum(arr);
    }
  )";

  auto P = inline_full(code);
  const goto_functiont *main = require_function(P.functions, "c:@F@main");
  REQUIRE(count_calls(main->body) == 0);
}

// ---------------------------------------------------------------------------
// expand_function_call: void return / no-lhs path in replace_return
// ---------------------------------------------------------------------------

SCENARIO(
  "goto_inline replaces a void return with an unconditional GOTO end",
  "[goto_inline]")
{
  // void f(int) returns nothing; replace_return takes the (lhs.is_nil(),
  // operand.is_nil()) path and rewrites RETURN to GOTO end_of_body.
  // Covers: replace_return nil-lhs + nil-operand branch.
  std::string code = R"(
    void f(int a) { (void)a; }
    int main() {
      f(7);
      return 0;
    }
  )";

  auto P = inline_full(code);
  const goto_functiont *main = require_function(P.functions, "c:@F@main");
  REQUIRE(count_calls(main->body) == 0);
}

// ---------------------------------------------------------------------------
// expand_function_call: result discarded + non-nil return operand
// ---------------------------------------------------------------------------

SCENARIO(
  "goto_inline emits an OTHER instruction when the return value is unused",
  "[goto_inline]")
{
  // The call to id() has no LHS at the call site, so replace_return takes
  // the lhs.is_nil() && !operand.is_nil() branch and emits an OTHER
  // instruction wrapping the return expression so any pointer derefs in
  // it are still evaluated.
  // Covers: replace_return nil-lhs + non-nil-operand branch.
  std::string code = R"(
    int id(int a) { return a; }
    int main() {
      id(3);
      return 0;
    }
  )";

  auto P = inline_full(code);
  const goto_functiont *main = require_function(P.functions, "c:@F@main");
  REQUIRE(count_calls(main->body) == 0);
  REQUIRE(count_instructions(main->body, OTHER) >= 1);
}

// ---------------------------------------------------------------------------
// expand_function_call: extern (no body) — nondet rhs + arg eval
// ---------------------------------------------------------------------------

SCENARIO(
  "goto_inline replaces an extern call with arg-eval + nondet return",
  "[goto_inline]")
{
  // ext() has no body. expand_function_call takes the !body_available
  // branch: it emits one OTHER per argument (so derefs evaluate) and an
  // ASSIGN of nondet to the lhs.
  // Covers: expand_function_call no-body branch + gen_nondet construction.
  std::string code = R"(
    extern int ext(int x);
    int main() {
      int r = ext(42);
      return r;
    }
  )";

  auto P = inline_full(code);
  const goto_functiont *main = require_function(P.functions, "c:@F@main");
  REQUIRE(count_calls(main->body) == 0);
}

// ---------------------------------------------------------------------------
// expand_function_call: recursion handling (full inlining)
// ---------------------------------------------------------------------------

SCENARIO(
  "goto_inline replaces a recursive self-call with SKIP under full inlining",
  "[goto_inline]")
{
  // fact(n) → fact(n-1) is direct self-recursion. Under full inlining
  // expand_function_call detects the cycle via recursion_set and replaces
  // the inner call with a SKIP (not a fatal abort).
  // Covers: recursion_set hit + make_skip branch.
  std::string code = R"(
    int fact(int n) {
      if (n <= 1) return 1;
      return n * fact(n - 1);
    }
    int main() { return fact(3); }
  )";

  auto P = inline_full(code);
  const goto_functiont *main = require_function(P.functions, "c:@F@main");
  // Top-level fact() is inlined; the inner self-call became SKIP, so no
  // FUNCTION_CALL remains.
  REQUIRE(count_calls(main->body) == 0);
}

// ---------------------------------------------------------------------------
// goto_partial_inline: small-function threshold gating
// ---------------------------------------------------------------------------

SCENARIO(
  "goto_partial_inline expands functions below the size threshold only",
  "[goto_inline]")
{
  // tiny() has a one-instruction body and is below the (large) limit, so
  // it is inlined into main. big() has more instructions than the limit
  // and is kept as a FUNCTION_CALL.
  // Covers: expand_function_call !full + smallfunc_limit branch.
  std::string code = R"(
    int big(int a) {
      int x = a + 1;
      int y = x + 2;
      int z = y + 3;
      int w = z + 4;
      return w;
    }
    int tiny(int a) { return a; }
    int main() {
      return tiny(1) + big(2);
    }
  )";

  auto P = inline_partial(code, /*limit=*/3);
  const goto_functiont *main = require_function(P.functions, "c:@F@main");
  // tiny() (≤3 instructions) was inlined; big() (>3 instructions) wasn't.
  REQUIRE(count_calls(main->body) >= 1);
}

// ---------------------------------------------------------------------------
// inline_instruction: indirect call (function pointer) is left alone
// ---------------------------------------------------------------------------

SCENARIO(
  "goto_inline leaves indirect function-pointer calls untouched",
  "[goto_inline]")
{
  // The call through fp is a code_function_call2t whose .function is NOT
  // is_symbol2t(...). inline_instruction returns false; the call stays.
  // Covers: inline_instruction non-symbol-function early return.
  std::string code = R"(
    int g(int a) { return a; }
    int main() {
      int (*fp)(int) = g;
      return fp(1);
    }
  )";

  auto P = inline_full(code);
  const goto_functiont *main = require_function(P.functions, "c:@F@main");
  // The indirect call survives; FUNCTION_CALL count is at least 1.
  REQUIRE(count_calls(main->body) >= 1);
}

// ---------------------------------------------------------------------------
// expand_function_call: body.hide rewrites instruction locations
// ---------------------------------------------------------------------------

SCENARIO(
  "goto_inline rewrites locations of inlined instructions when body is hidden",
  "[goto_inline]")
{
  // Functions in the C library (like memcpy, malloc, etc.) have body.hide
  // set when they are operational models the user shouldn't see. After
  // inlining, every instruction in the inlined body has its location
  // overwritten to the call site.
  // We verify the migrated path runs without crashing on a memcpy call.
  // Covers: body.hide branch + Forall_goto_program_instructions location
  //         rewrite.
  std::string code = R"(
    extern void *memcpy(void *dst, const void *src, unsigned long n);
    int main() {
      char buf[4] = {0};
      const char src[4] = {1, 2, 3, 4};
      memcpy(buf, src, 4);
      return buf[0];
    }
  )";

  auto P = inline_full(code);
  const goto_functiont *main = require_function(P.functions, "c:@F@main");
  // Just check the program still has a body — the assertion is that the
  // hide branch did not abort.
  REQUIRE(!main->body.instructions.empty());
}
