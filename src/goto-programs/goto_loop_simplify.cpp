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

/// Body-shape predicate. @p body_first is the first instruction *after*
/// loop_head; @p body_last is the back-edge instruction (still inclusive).
/// Returns true and fills @p modified iff every instruction in
/// [body_first, body_last) is one of:
///   - LOCATION / SKIP / DEAD
///   - ASSIGN to a non-pointer symbol2t
/// AND body_last is an unconditional GOTO to loop_head (the back-edge).
/// Anything else (calls, asserts, assumes, escaping gotos, pointer writes)
/// disqualifies the loop.
bool body_is_safe(
  inst_iter body_first,
  inst_iter body_last,
  inst_iter loop_head,
  name_set &modified)
{
  // The back-edge itself must be an unconditional GOTO to loop_head.
  if (!body_last->is_goto())
    return false;
  if (body_last->targets.size() != 1)
    return false;
  if (*body_last->targets.begin() != loop_head)
    return false;
  if (!is_true(body_last->guard))
    return false;

  for (inst_iter it = body_first; it != body_last; ++it)
  {
    if (it->is_location() || it->is_skip() || it->is_decl())
      continue;

    if (it->type == DEAD)
    {
      // A var declared and killed inside the body never escapes the loop;
      // drop it from the modified set so the post-loop liveness check
      // doesn't expect a matching DEAD after exit.
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
      // code_dead2t holds the victim's name in `code->value` (an irep_idt)
      // — collect_symbol_names won't see it because there's no symbol2t
      // operand. Pull the name directly via the type-specific accessor.
      const code_dead2t &cd = to_code_dead2t(it->code);
      alive.erase(cd.value);
      continue;
    }

    // Any read of a still-alive modified var disqualifies the rewrite.
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

/// Rewrite a removable loop in place: head IF -> SKIP, body -> SKIP each,
/// back-edge GOTO -> SKIP. Targets to loop_head from outside the loop
/// (which we've ruled out by body_is_safe — but back-edges in nested loops
/// aren't covered here, see TODO) become targets to a SKIP, which
/// remove_no_op forwards.
void erase_loop(inst_iter loop_head, inst_iter loop_exit)
{
  for (inst_iter it = loop_head; it != std::next(loop_exit); ++it)
    it->make_skip();
}

/// One pass over a function. Returns true iff any loop was removed (so the
/// caller can iterate to fixpoint).
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
    inst_iter loop_exit = it; // back-edge instruction

    // Self-loops are already handled by goto_loopst::find_function_loops's
    // make_assumption rewrite. Skip them here.
    if (loop_head == loop_exit)
      continue;

    // loop_head must be a forward IF to past the loop (the exit guard).
    if (!loop_head->is_goto() || loop_head->targets.size() != 1)
      continue;
    inst_iter loop_after_exit = *loop_head->targets.begin();

    // Body is everything between loop_head (exclusive) and loop_exit
    // (inclusive — the back-edge GOTO).
    inst_iter body_first = std::next(loop_head);

    name_set modified;
    if (!body_is_safe(body_first, loop_exit, loop_head, modified))
      continue;

    if (!modified_vars_die_immediately(
          loop_after_exit, body.instructions.end(), modified))
      continue;

    erase_loop(loop_head, loop_exit);
    changed = true;

    // Iterator is now pointing at a SKIP'd back-edge; advance and
    // continue. The `for` loop's `++it` will move us past it.
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
