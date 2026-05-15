#include <goto-programs/goto_loop_simplify.h>
#include <goto-programs/loopst.h>
#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/config.h>
#include <unordered_set>

namespace
{
using inst_iter = goto_programt::instructionst::iterator;
using name_set = std::unordered_set<irep_idt, irep_id_hash>;

/// Collect names of every plain symbol2t reachable in @p expr.
void collect_symbol_names(const expr2tc &expr, name_set &out)
{
  if (is_nil_expr(expr))
    return;
  if (is_symbol2t(expr))
    out.insert(to_symbol2t(expr).thename);
  expr->foreach_operand(
    [&out](const expr2tc &e) { collect_symbol_names(e, out); });
}

/// Body-shape predicate. @p body_first is the first instruction inside
/// the loop body; @p body_last is the back-edge instruction. Returns
/// true and fills @p modified with the names of every variable assigned
/// in the body iff every instruction in [body_first, body_last) is one of:
///   - LOCATION / SKIP / DECL / DEAD
///   - ASSIGN to a non-pointer symbol2t
/// AND body_last is a backwards GOTO targetting @p loop_head. Anything
/// else (calls, asserts, assumes, escaping gotos, pointer writes)
/// disqualifies the loop.
///
/// The back-edge may be unconditional (for/while shape) or conditional
/// (do-while shape). We don't care which here — shape discrimination is
/// the caller's job.
bool body_is_safe(
  inst_iter body_first,
  inst_iter body_last,
  inst_iter loop_head,
  name_set &modified)
{
  if (!body_last->is_backwards_goto())
    return false;
  if (body_last->targets.size() != 1)
    return false;
  if (*body_last->targets.begin() != loop_head)
    return false;

  for (inst_iter it = body_first; it != body_last; ++it)
  {
    if (it->is_location() || it->is_skip() || it->is_decl())
      continue;

    if (it->type == DEAD)
    {
      // A var declared and killed inside the body never escapes the loop.
      modified.erase(to_code_dead2t(it->code).value);
      continue;
    }

    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      if (!is_symbol2t(a.target))
        return false;
      if (is_pointer_type(a.target))
        return false;
      modified.insert(to_symbol2t(a.target).thename);
      continue;
    }

    // Anything else — function call, assert, assume, nested goto, etc. —
    // makes the loop observable.
    return false;
  }

  return true;
}

/// Liveness check after the loop. Walk forward from @p after_exit and
/// confirm every name in @p modified has a DEAD instruction before any
/// read. Narrow shape: the next non-LOCATION/SKIP instructions must be
/// DEAD operations on the modified vars (in any order) until each has
/// been seen. Any other instruction kind, or a read of a still-live
/// modified var, refuses.
bool modified_vars_die_immediately(
  inst_iter after_exit,
  inst_iter end,
  const name_set &modified)
{
  name_set alive(modified);

  for (inst_iter it = after_exit; it != end && !alive.empty(); ++it)
  {
    if (it->is_location() || it->is_skip())
      continue;

    if (it->type == DEAD)
    {
      alive.erase(to_code_dead2t(it->code).value);
      continue;
    }

    name_set read_names;
    collect_symbol_names(it->code, read_names);
    collect_symbol_names(it->guard, read_names);
    for (const irep_idt &n : read_names)
      if (alive.count(n))
        return false;

    // The instruction is not LOCATION / SKIP / DEAD, so the narrow-shape
    // requirement isn't met. Refuse conservatively.
    return false;
  }

  return alive.empty();
}

/// Rewrite a removable loop in place: every instruction from loop_head
/// through loop_exit becomes SKIP.
void erase_loop(inst_iter loop_head, inst_iter loop_exit)
{
  for (inst_iter it = loop_head; it != std::next(loop_exit); ++it)
    it->make_skip();
}

/// Information extracted from a recognised counter loop.
struct counter_loop_info
{
  enum relation_kind
  {
    LT,
    LE,
    GT,
    GE
  };

  expr2tc induction_sym; // original symbol2tc (for the post-value ASSIGN)
  BigInt init;
  BigInt bound;
  BigInt step;        // positive integer; sign is encoded in step_negative
  bool step_negative; // true if i = i - step (decrementing loop)
  relation_kind rel;
  type2tc type;
};

/// Negate a relation kind: `<` ↔ `>=`, `<=` ↔ `>`.
counter_loop_info::relation_kind
negate_relation(counter_loop_info::relation_kind r)
{
  switch (r)
  {
  case counter_loop_info::LT:
    return counter_loop_info::GE;
  case counter_loop_info::LE:
    return counter_loop_info::GT;
  case counter_loop_info::GT:
    return counter_loop_info::LE;
  case counter_loop_info::GE:
    return counter_loop_info::LT;
  }
  return counter_loop_info::LT; // unreachable
}

/// Try to parse @p guard as the EXIT condition of a counter loop.
/// Accepts both shapes the frontend / interval analysis produce:
///   - `!(REL(i, c))` — original IF !cond form (cond is the
///     continuation, the inner REL is the continuation relation).
///   - `REL(i, c)` — interval-analysis-rewritten form (REL is the
///     exit relation directly).
///
/// Stores the CONTINUATION relation in @p info.rel (so the post-value
/// formulas can stay phrased in terms of "iterate while cond").
bool parse_guard(const expr2tc &guard, counter_loop_info &info)
{
  expr2tc cond;
  bool inner_is_continuation;
  if (is_not2t(guard))
  {
    cond = to_not2t(guard).value;
    inner_is_continuation = true;
  }
  else
  {
    cond = guard;
    inner_is_continuation = false;
  }

  expr2tc lhs, rhs;
  if (is_lessthan2t(cond))
  {
    lhs = to_lessthan2t(cond).side_1;
    rhs = to_lessthan2t(cond).side_2;
    info.rel = counter_loop_info::LT;
  }
  else if (is_lessthanequal2t(cond))
  {
    lhs = to_lessthanequal2t(cond).side_1;
    rhs = to_lessthanequal2t(cond).side_2;
    info.rel = counter_loop_info::LE;
  }
  else if (is_greaterthan2t(cond))
  {
    lhs = to_greaterthan2t(cond).side_1;
    rhs = to_greaterthan2t(cond).side_2;
    info.rel = counter_loop_info::GT;
  }
  else if (is_greaterthanequal2t(cond))
  {
    lhs = to_greaterthanequal2t(cond).side_1;
    rhs = to_greaterthanequal2t(cond).side_2;
    info.rel = counter_loop_info::GE;
  }
  else
    return false;

  // If the parsed relation is the exit form, negate it to recover the
  // continuation form expected downstream.
  if (!inner_is_continuation)
    info.rel = negate_relation(info.rel);

  // One side must be the induction var (a symbol2t), the other a
  // constant_int2t. Constant on the right is the canonical shape; if
  // it's on the left, flip the relation.
  if (is_symbol2t(lhs) && is_constant_int2t(rhs))
  {
    info.induction_sym = lhs;
    info.bound = to_constant_int2t(rhs).value;
  }
  else if (is_constant_int2t(lhs) && is_symbol2t(rhs))
  {
    info.induction_sym = rhs;
    info.bound = to_constant_int2t(lhs).value;
    // Flip: `c < i` ≡ `i > c`, etc.
    switch (info.rel)
    {
    case counter_loop_info::LT:
      info.rel = counter_loop_info::GT;
      break;
    case counter_loop_info::LE:
      info.rel = counter_loop_info::GE;
      break;
    case counter_loop_info::GT:
      info.rel = counter_loop_info::LT;
      break;
    case counter_loop_info::GE:
      info.rel = counter_loop_info::LE;
      break;
    }
  }
  else
    return false;

  if (!is_bv_type(info.induction_sym))
    return false;
  info.type = info.induction_sym->type;
  return true;
}

/// Find the single ASSIGN to the induction var in [body_first, body_last).
/// Must be of the form `i = i + STEP` or `i = i - STEP` with STEP a
/// constant_int. Other ASSIGNs in the body disqualify (would have other
/// modified vars). Returns true on success and fills the step fields.
bool parse_step(
  inst_iter body_first,
  inst_iter body_last,
  const irep_idt &induction_name,
  counter_loop_info &info)
{
  bool found = false;
  for (inst_iter it = body_first; it != body_last; ++it)
  {
    if (it->is_location() || it->is_skip() || it->is_decl() || it->type == DEAD)
      continue;
    if (!it->is_assign())
      return false;

    const code_assign2t &a = to_code_assign2t(it->code);
    if (!is_symbol2t(a.target))
      return false;
    if (to_symbol2t(a.target).thename != induction_name)
      return false;

    // RHS must be `i + const` or `i - const` (induction var first).
    expr2tc rhs_lhs, rhs_rhs;
    bool is_sub;
    if (is_add2t(a.source))
    {
      rhs_lhs = to_add2t(a.source).side_1;
      rhs_rhs = to_add2t(a.source).side_2;
      is_sub = false;
    }
    else if (is_sub2t(a.source))
    {
      rhs_lhs = to_sub2t(a.source).side_1;
      rhs_rhs = to_sub2t(a.source).side_2;
      is_sub = true;
    }
    else
      return false;

    // Canonical form: induction var on the left, constant on the right.
    // Symmetric for add only (sub is non-commutative).
    if (is_symbol2t(rhs_lhs) && is_constant_int2t(rhs_rhs))
    {
      if (to_symbol2t(rhs_lhs).thename != induction_name)
        return false;
    }
    else if (!is_sub && is_constant_int2t(rhs_lhs) && is_symbol2t(rhs_rhs))
    {
      if (to_symbol2t(rhs_rhs).thename != induction_name)
        return false;
      std::swap(rhs_lhs, rhs_rhs);
    }
    else
      return false;

    BigInt step_val = to_constant_int2t(rhs_rhs).value;
    if (step_val.is_zero())
      return false;
    if (step_val.is_negative())
    {
      // i = i + (-k) is the same as i = i - k.
      step_val.negate();
      is_sub = !is_sub;
    }
    info.step = step_val;
    info.step_negative = is_sub;

    if (found)
      return false; // multiple non-trivial assigns — give up
    found = true;
  }
  return found;
}

/// Walk backwards from @p loop_head looking for `ASSIGN i = INIT` where
/// INIT is a constant_int. Skips LOCATION/SKIP/DECL/DEAD. Stops at the
/// first non-trivial instruction. Returns true on success.
bool parse_init(
  inst_iter begin,
  inst_iter loop_head,
  const irep_idt &induction_name,
  counter_loop_info &info)
{
  if (loop_head == begin)
    return false;
  inst_iter it = loop_head;
  while (it != begin)
  {
    --it;
    // A goto-target instruction between begin and loop_head means
    // another CFG predecessor can reach loop_head without flowing
    // through our textually-nearest init ASSIGN. Committing to the
    // init we'd find here would silently drop the other path's value.
    // Bail to keep the rewrite sound.
    if (it->is_target())
      return false;
    if (it->is_location() || it->is_skip() || it->is_decl() || it->type == DEAD)
      continue;
    if (!it->is_assign())
      return false;
    const code_assign2t &a = to_code_assign2t(it->code);
    if (!is_symbol2t(a.target))
      return false;
    if (to_symbol2t(a.target).thename != induction_name)
      return false;
    if (!is_constant_int2t(a.source))
      return false;
    info.init = to_constant_int2t(a.source).value;
    return true;
  }
  return false;
}

/// Compute the post-loop value of the induction variable from a recognized
/// counter loop. Returns std::nullopt if no iterations happen or the
/// computation would overflow `info.type`.
std::optional<BigInt> compute_post_value(const counter_loop_info &c)
{
  // Decrementing loop: only handle GT/GE for now (canonical `for(i=N-1;
  // i>=0; i--)`). Symmetric to increment via reflection.
  // To keep v1 narrow: only handle `i < bound` with step > 0 and
  // `i > bound` with step < 0.
  if (!c.step_negative && c.rel == counter_loop_info::LT)
  {
    if (c.init >= c.bound)
      return c.init; // zero iterations
    BigInt span = c.bound - c.init;
    // ceiling division: (span + step - 1) / step
    BigInt n = (span + c.step - 1) / c.step;
    return c.init + n * c.step;
  }
  if (!c.step_negative && c.rel == counter_loop_info::LE)
  {
    if (c.init > c.bound)
      return c.init;
    // Exit when i > bound, i.e. i >= bound+1. Reduce to LT case.
    BigInt target = c.bound + 1;
    BigInt span = target - c.init;
    BigInt n = (span + c.step - 1) / c.step;
    return c.init + n * c.step;
  }
  if (c.step_negative && c.rel == counter_loop_info::GT)
  {
    if (c.init <= c.bound)
      return c.init;
    BigInt span = c.init - c.bound;
    BigInt n = (span + c.step - 1) / c.step;
    return c.init - n * c.step;
  }
  if (c.step_negative && c.rel == counter_loop_info::GE)
  {
    if (c.init < c.bound)
      return c.init;
    BigInt target = c.bound - 1;
    BigInt span = c.init - target;
    BigInt n = (span + c.step - 1) / c.step;
    return c.init - n * c.step;
  }
  return std::nullopt; // unhandled relation/sign combo
}

/// Try to rewrite a counter-shaped loop as a single `i = post_value`
/// assignment. Returns true on success and mutates @p body.
///
/// @p exit_guard is `!cond` — the condition under which control falls
/// through past the loop. The caller normalizes this so we don't need
/// to discriminate the IR shape (for/while loop_head's guard is
/// `!cond` already; do-while back-edge's guard is `cond` and the
/// caller negates).
///
/// @p is_dowhile says whether the loop runs its body unconditionally
/// once before the first guard test. Affects compute_post_value: for
/// do-while we step `init` by `step` before applying the standard
/// formula, modelling the at-least-one-iteration semantics.
bool try_step_recognition(
  goto_programt &body,
  inst_iter loop_head,
  inst_iter loop_exit,
  inst_iter body_first,
  const expr2tc &exit_guard,
  bool is_dowhile)
{
  counter_loop_info info;
  if (!parse_guard(exit_guard, info))
    return false;
  const irep_idt induction_name = to_symbol2t(info.induction_sym).thename;
  if (!parse_step(body_first, loop_exit, induction_name, info))
    return false;
  if (!parse_init(body.instructions.begin(), loop_head, induction_name, info))
    return false;

  // do-while runs the body once unconditionally — fold one step of the
  // induction var into init before applying the standard formula. After
  // this, the rest of the iteration is identical to for/while semantics.
  if (is_dowhile)
  {
    if (info.step_negative)
      info.init -= info.step;
    else
      info.init += info.step;
  }

  std::optional<BigInt> post = compute_post_value(info);
  if (!post)
    return false;

  // Verify the post value fits in the induction var's type. integer2binary
  // would assert on values that don't fit; check by re-encoding and
  // comparing.
  BigInt fitted = *post;
  // For bit-vector types, range is [type_min, type_max]. arith_tools
  // exposes is_type_min/is_type_max; simplest check: round-trip through
  // from_integer and ensure no truncation. from_integer is permissive,
  // so we re-derive the bit width and check magnitude.
  const unsigned width = info.type->get_width();
  BigInt bound_hi;
  BigInt bound_lo;
  if (is_signedbv_type(info.type))
  {
    bound_hi = BigInt(1) << (width - 1);
    --bound_hi;               // 2^(w-1) - 1
    bound_lo = -bound_hi - 1; // -2^(w-1)
  }
  else
  {
    bound_hi = (BigInt(1) << width) - 1;
    bound_lo = 0;
  }
  if (fitted < bound_lo || fitted > bound_hi)
    return false;

  // Build the rewrite fragment: ASSIGN i = post_value.
  goto_programt fragment;
  inst_iter t = fragment.add_instruction(ASSIGN);
  t->code =
    code_assign2tc(info.induction_sym, constant_int2tc(info.type, fitted));
  t->location = loop_head->location;

  body.insert_swap(loop_head, fragment);

  // loop_head iterator now points to the inserted ASSIGN; the original
  // IF lives at std::next(loop_head). SKIP the IF, body, and back-edge.
  for (inst_iter sk = std::next(loop_head); sk != std::next(loop_exit); ++sk)
    sk->make_skip();
  return true;
}

/// One pass over a function. Returns true iff any loop was erased (so
/// the caller can iterate to fixpoint).
bool simplify_function_once(goto_functiont &fn)
{
  if (!fn.body_available)
    return false;

  goto_programt &body = fn.body;
  bool changed = false;

  for (inst_iter it = body.instructions.begin(); it != body.instructions.end();
       ++it)
  {
    if (!it->is_backwards_goto())
      continue;
    if (it->targets.size() != 1)
      continue;

    inst_iter loop_head = *it->targets.begin();
    inst_iter loop_exit = it;

    // Self-loops `A: IF cond GOTO A` / `A: GOTO A` — rewrite to ASSUME
    // in place. Mirrors goto_loopst::find_function_loops's rewrite,
    // which only fires when a loop-discovering pass is enabled
    // (interval-analysis, k-induction, loop-invariant, contractor).
    // Doing it here makes the rewrite unconditional for plain BMC.
    // symex_goto.cpp's self-loop handler stays as a defensive fallback
    // for modes that skip this pass (--termination, --unwinding-
    // assertions).
    if (loop_head == loop_exit)
    {
      if (is_true(it->guard))
      {
        // Unconditional infinite loop: kill the path.
        it->make_assumption(gen_false_expr());
      }
      else
      {
        // `IF cond GOTO self` exits exactly when !cond; the post-loop
        // state is constrained by !cond. Copy the guard out before
        // make_assumption (clear() runs first, resetting it).
        expr2tc cond = it->guard;
        make_not(cond);
        it->make_assumption(cond);
      }
      changed = true;
      continue;
    }

    // Two loop shapes the C/C++ frontend produces:
    //
    // for/while:
    //   loop_head: IF !cond GOTO E        (exit-test up front)
    //              body
    //              GOTO loop_head         (unconditional back-edge)
    //   E:
    //
    // do-while:
    //   loop_head: body                   (no exit-test at top)
    //              IF cond GOTO loop_head (conditional back-edge)
    //
    // Discrimination: if loop_head is a forward GOTO, it's the for/while
    // exit IF — body starts after it, post-loop code starts at the IF's
    // target. Otherwise it's do-while — body starts at loop_head, post-
    // loop code starts after the back-edge.
    inst_iter body_first;
    inst_iter after_loop;
    expr2tc exit_guard;
    bool is_dowhile;
    if (loop_head->is_goto() && !loop_head->is_backwards_goto())
    {
      if (loop_head->targets.size() != 1)
        continue;
      body_first = std::next(loop_head);
      after_loop = *loop_head->targets.begin();
      // The IF's guard IS the exit condition (semantically: "GOTO E
      // if true"). Shape: typically `!(cond)` for non-simplified
      // input, or the un-negated exit relation (`i >= N`) when
      // interval analysis already rewrote it. parse_guard handles
      // both.
      exit_guard = loop_head->guard;
      is_dowhile = false;
    }
    else
    {
      body_first = loop_head;
      after_loop = std::next(loop_exit);
      // Back-edge's guard is the continuation `cond` (jump back if
      // true). Negate for the exit condition. make_not strips an
      // outer `not` if present so the result stays canonical.
      exit_guard = loop_exit->guard;
      make_not(exit_guard);
      is_dowhile = true;
    }

    // Under --termination: if the exit guard simplifies to constant
    // false, the loop's only exit edge is never taken — `while(1) { }`,
    // `for(;;) { }`. Rewrite the loop head to `assume(false)` so the
    // path is killed (correct model of a non-terminating execution: no
    // post-loop state is ever reached). The termination reduction then
    // sees the assert(false) marker as unreachable, IS reports UNSAT,
    // and we conclude non-termination.
    //
    // Gated to --termination only: under default BMC the unwinding
    // assertion is the user-visible signal of non-termination, and
    // collapsing the loop to assume(false) would silently suppress it.
    // (See body_is_safe-independent check above — we DO want this to
    // fire even when the body is non-trivial, as long as the exit guard
    // is constant false, because under --termination "no exit edge"
    // dominates "body has side effects".)
    if (config.options.get_bool_option("termination"))
    {
      expr2tc exit_guard_simp = exit_guard;
      simplify(exit_guard_simp);
      if (is_false(exit_guard_simp))
      {
        loop_head->make_assumption(gen_false_expr());
        erase_loop(std::next(loop_head), loop_exit);
        changed = true;
      }
      // Under --termination, ONLY the assume(false) rewrite is sound:
      // Path 1 erasure and Path 2 step recognition would also silently
      // hide loops whose presence matters for the termination reduction.
      continue;
    }

    name_set modified;
    if (!body_is_safe(body_first, loop_exit, loop_head, modified))
      continue;

    // Path 1: empty-body top-test loop whose modified set is empty
    // (vacuously dying) — rewrite the head to assume(exit_guard) and
    // SKIP the rest. Sound because an empty body cannot change the
    // exit guard's truth value, so the value at exit equals the value
    // at entry. Non-empty bodies (even if their modified vars die)
    // are NOT erased here: the loop's termination and post-state are
    // both observable, and a syntactic "vars die" check proves
    // neither. Counter patterns get the precise rewrite via Path 2.
    if (
      modified.empty() && !is_dowhile &&
      modified_vars_die_immediately(
        after_loop, body.instructions.end(), modified))
    {
      loop_head->make_assumption(exit_guard);
      erase_loop(std::next(loop_head), loop_exit);
      changed = true;
      continue;
    }

    // Path 2: vars escape, but the loop may be a recognizable counter
    // pattern (`for(i=INIT; i<BOUND; i++)` with constants) — derive the
    // exact post-value and rewrite to `i = post_value`. Strictly precise:
    // unlike havoc+assume, this preserves the assertion `i == N` after
    // a `for(i=0;i<N;i++);` loop.
    //
    // Interval analysis has already run, so symbolic bounds proven to
    // be singletons are now constant_int2t.
    if (try_step_recognition(
          body, loop_head, loop_exit, body_first, exit_guard, is_dowhile))
    {
      changed = true;
      continue;
    }
  }

  return changed;
}

void simplify_function(goto_functiont &fn)
{
  while (simplify_function_once(fn))
    ;
}
} // namespace

void goto_loop_simplify(goto_functionst &goto_functions)
{
  // Gated on the user explicitly opting out of the unwinding-
  // assertion signal (--no-unwinding-assertions) or opting into the
  // termination reduction (--termination). Outside these two modes
  // the pass is a no-op.
  //
  // Why the gate matters: by default, ESBMC emits an unwinding
  // assertion at every loop back-edge that fires when the loop
  // hasn't fully unwound within --unwind k. That assertion is the
  // user-visible non-termination signal at the chosen depth.
  // Erasing or rewriting a loop in goto_loop_simplify would suppress
  // that signal even when the loop body is otherwise side-effect-
  // free, breaking the verification contract (cf. regression/
  // bitwuzla/get_model_values, regression/parallel-solving/uthash-1).
  //
  // What each mode does:
  //   --no-unwinding-assertions: Path 1 (erase dead loops with no
  //     escaping side effects) and Path 2 (step recognition for
  //     constant-bound counter loops) run. The self-loop and
  //     constant-false-exit-guard rewrites stay off — symex's loop
  //     handling already does the right thing in that mode.
  //
  //   --termination: only the constant-false-exit-guard rewrite (and
  //     the single-instruction self-loop rewrite) fire — both gated
  //     inside simplify_function_once. Path 1 and Path 2 are skipped
  //     because they would discard loops whose presence is observable
  //     under the termination reduction.
  //
  // Coverage modes: also skipped — erasing loops drops branch points
  // the coverage instrumentation needs to count.
  if (
    !config.options.get_bool_option("no-unwinding-assertions") &&
    !config.options.get_bool_option("termination"))
    return;

  if (
    config.options.get_bool_option("assertion-coverage") ||
    config.options.get_bool_option("assertion-coverage-claims") ||
    config.options.get_bool_option("branch-coverage") ||
    config.options.get_bool_option("branch-coverage-claims") ||
    config.options.get_bool_option("branch-function-coverage") ||
    config.options.get_bool_option("branch-function-coverage-claims") ||
    config.options.get_bool_option("condition-coverage") ||
    config.options.get_bool_option("condition-coverage-claims") ||
    config.options.get_bool_option("condition-coverage-rm") ||
    config.options.get_bool_option("condition-coverage-claims-rm") ||
    config.options.get_bool_option("k-path-coverage") ||
    config.options.get_bool_option("k-path-coverage-claims"))
    return;

  Forall_goto_functions (it, goto_functions)
    simplify_function(it->second);

  goto_functions.update();
}
