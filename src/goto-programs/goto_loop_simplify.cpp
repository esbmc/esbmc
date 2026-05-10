#include <goto-programs/goto_loop_simplify.h>
#include <goto-programs/loopst.h>
#include <irep2/irep2_utils.h>
#include <util/config.h>
#include <unordered_map>
#include <unordered_set>

namespace
{
using inst_iter = goto_programt::instructionst::iterator;
using name_set = std::unordered_set<irep_idt, irep_id_hash>;
/// Maps a modified symbol's name to the first symbol2tc expression that
/// referenced it inside the loop body. We need the expression (not just
/// the name) to build the havoc ASSIGN, which requires the symbol2t's
/// full state (renaming level, num fields, type, etc.) — reconstructing
/// from name + type alone is fragile.
using modified_map = std::unordered_map<irep_idt, expr2tc, irep_id_hash>;

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
/// true and fills @p modified iff every instruction in
/// [body_first, body_last) is one of:
///   - LOCATION / SKIP / DECL / DEAD
///   - ASSIGN to a non-pointer symbol2t
/// AND body_last is a backwards GOTO targetting @p loop_head. Anything
/// else (calls, asserts, assumes, escaping gotos, pointer writes)
/// disqualifies the loop.
///
/// The back-edge may be unconditional (for/while shape) or conditional
/// (do-while shape). We don't care which here — the caller picks the
/// right exit guard.
bool body_is_safe(
  inst_iter body_first,
  inst_iter body_last,
  inst_iter loop_head,
  modified_map &modified)
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
      // Insert preserves the first symbol2tc seen for each name. We keep
      // the original (pre-renaming) expression so havoc can rebuild a
      // valid ASSIGN.
      modified.emplace(to_symbol2t(a.target).thename, a.target);
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
  const modified_map &modified)
{
  name_set alive;
  for (const auto &kv : modified)
    alive.insert(kv.first);

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

/// Rewrite a side-effecting empty-body loop in place with the havoc +
/// assume(exit_cond) pattern. The havoc gives each modified var an
/// unconstrained value; the assume constrains them to satisfy the exit
/// condition. Sound for safety BMC because the loop's only effect on
/// post-loop state was setting these vars to a value satisfying the
/// exit condition — exactly what we now express.
///
/// Works uniformly for while/for (loop_head is the exit IF, exit_guard
/// is its guard `!cond`) and do-while (loop_head is the body, exit_guard
/// is the negation of the back-edge's guard `cond`).
void rewrite_loop_with_havoc(
  goto_programt &program,
  inst_iter loop_head,
  inst_iter loop_exit,
  const expr2tc &exit_guard,
  const modified_map &modified)
{
  locationt loc = loop_head->location;

  // Build the fragment: havocs + ASSUME(exit_cond). insert_swap places
  // it AT loop_head's position, pushing the original instructions down.
  // After insert_swap, the loop_head iterator points to the first
  // fragment instruction; the original head lives at
  // std::next(loop_head, fragment_size).
  goto_programt fragment;
  for (const auto &kv : modified)
  {
    const expr2tc &sym = kv.second;
    inst_iter t = fragment.add_instruction(ASSIGN);
    t->code = code_assign2tc(sym, gen_nondet(sym->type));
    t->location = loc;
  }
  {
    inst_iter t = fragment.add_instruction(ASSUME);
    t->guard = exit_guard;
    t->location = loc;
  }

  const size_t fragment_size = fragment.instructions.size();
  program.insert_swap(loop_head, fragment);

  // SKIP the original head, body, and back-edge. loop_exit's iterator
  // is stable across insert_swap.
  inst_iter original_head = loop_head;
  for (size_t i = 0; i < fragment_size; ++i)
    ++original_head;
  for (inst_iter it = original_head; it != std::next(loop_exit); ++it)
    it->make_skip();
}

/// One pass over a function. Returns true iff any loop was rewritten (so
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

    // Self-loops are already handled by goto_loopst::find_function_loops's
    // make_assumption rewrite. Skip them here.
    if (loop_head == loop_exit)
      continue;

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
    // exit IF — body starts after it, exit condition is loop_head's
    // guard (already `!cond`). Otherwise it's do-while — body starts at
    // loop_head, exit condition is the negation of the back-edge's guard.
    inst_iter body_first;
    inst_iter after_loop;
    expr2tc exit_guard;
    if (loop_head->is_goto() && !loop_head->is_backwards_goto())
    {
      // for/while shape.
      if (loop_head->targets.size() != 1)
        continue;
      body_first = std::next(loop_head);
      after_loop = *loop_head->targets.begin();
      exit_guard = loop_head->guard; // already !cond
    }
    else
    {
      // do-while shape: the back-edge IS the conditional. The exit
      // condition is the negation of the back-edge's guard.
      body_first = loop_head;
      after_loop = std::next(loop_exit);
      exit_guard = loop_exit->guard;
      make_not(exit_guard);
    }

    modified_map modified;
    if (!body_is_safe(body_first, loop_exit, loop_head, modified))
      continue;

    // Path 1: every modified var dies immediately after the loop. The
    // loop has no observable effect; erase it entirely.
    if (modified_vars_die_immediately(
          after_loop, body.instructions.end(), modified))
    {
      erase_loop(loop_head, loop_exit);
      changed = true;
      continue;
    }

    // Path 2: vars escape. The loop's post-state matters, but its body
    // is still empty-body-shaped (only ASSIGNs to non-pointer locals,
    // no calls/asserts/assumes). Replace with havoc + assume(exit-cond).
    //
    // The havoc gives each modified var any value the type allows; the
    // assume constrains them to satisfy the exit condition. Subsequent
    // code sees a sound (possibly weaker) view of the post-loop state.
    //
    // Bails are already enforced by body_is_safe: no pointer-typed
    // modified vars (avoids alias-unsoundness from havoc'ing pointers),
    // no observable side effects in the body.
    if (modified.empty())
      continue; // nothing to havoc; nothing useful left to do

    rewrite_loop_with_havoc(body, loop_head, loop_exit, exit_guard, modified);
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
