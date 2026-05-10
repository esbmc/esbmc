#include <goto-programs/goto_loop_simplify.h>
#include <goto-programs/loopst.h>
#include <irep2/irep2_utils.h>
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

/// One pass over a function. Returns true iff any loop was erased (so
/// the caller can iterate to fixpoint).
bool simplify_function_once(goto_functiont &fn)
{
  if (!fn.body_available)
    return false;

  goto_programt &body = fn.body;
  bool changed = false;

  for (inst_iter it = body.instructions.begin();
       it != body.instructions.end();
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
    if (loop_head->is_goto() && !loop_head->is_backwards_goto())
    {
      if (loop_head->targets.size() != 1)
        continue;
      body_first = std::next(loop_head);
      after_loop = *loop_head->targets.begin();
    }
    else
    {
      body_first = loop_head;
      after_loop = std::next(loop_exit);
    }

    name_set modified;
    if (!body_is_safe(body_first, loop_exit, loop_head, modified))
      continue;

    // Erase only when every modified var dies immediately after the loop
    // — the loop has no observable effect on post-loop state. Loops whose
    // modified vars escape are left alone: a havoc-and-assume rewrite
    // would over-approximate the post-state (e.g. losing `i == N` for
    // `for(i=0;i<N;i++) ;`), turning provable assertions into spurious
    // failures, and could introduce unsoundness via array-index havoc
    // (e.g. strlen's `s[len]` guard with havoc'd len).
    if (!modified_vars_die_immediately(
          after_loop, body.instructions.end(), modified))
      continue;

    erase_loop(loop_head, loop_exit);
    changed = true;
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
  // Skipped under --termination or --unwinding-assertions: in those modes
  // an empty loop's presence is itself observable (it can fail to
  // terminate, or violate the unwind assertion).
  if (
    config.options.get_bool_option("termination") ||
    config.options.get_bool_option("unwinding-assertions"))
    return;

  Forall_goto_functions (it, goto_functions)
    simplify_function(it->second);

  goto_functions.update();
}
