#include <goto-programs/goto_k_induction.h>
#include <goto-programs/goto_loops.h>
#include <goto-programs/loopst.h>
#include <goto-programs/remove_no_op.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_guard.h>
#include <pointer-analysis/value_set.h>
#include <pointer-analysis/value_set_analysis.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/std_expr.h>
#include <memory>
#include <unordered_map>

namespace
{
using guardst = std::unordered_map<unsigned, guard2tc>;

/// Cached result of expanding a forward GOTO branch during the entry-
/// condition collection. The boolean is the recursion's return value at
/// this branch (`false_branch && true_branch`, i.e. true iff both
/// subbranches reach the loop end); the guardst is the set of guards
/// that should be merged into the caller's local guardst when this
/// cache entry fires. Storing only the boolean (the legacy design)
/// silently dropped these guards on every cache hit, weakening the
/// entry-condition assume.
struct branch_cache_entryt
{
  bool reaches;
  guardst guards_to_merge;
};
using marked_branchst = std::unordered_map<unsigned, branch_cache_entryt>;

/// Walk the loop body and collect, into @p guards, the conditions under
/// which control flows from @p loop_head to @p loop_exit. Used by
/// transform_loop to derive the entry condition of the loop (the
/// conjunction of every IF-branch guard taken along a body-reaching
/// path). @p cache is a loop-scoped memoisation table keyed on each
/// IF's location_number; cleared by transform_loop at the start of
/// every loop.
bool get_entry_cond_rec(
  const goto_programt::targett &loop_head,
  const goto_programt::targett &loop_exit,
  guardst &guards,
  marked_branchst &cache)
{
  // Let's walk the loop and collect the constraints to enter the
  // loop. This might be messy because of side-effects

  // entry and exit numbers
  auto const &entry_number = loop_head->location_number;
  auto const &exit_number = loop_exit->location_number;

  // We jumped outside the loop, don't collect this constraint
  if (entry_number > exit_number)
    return true;

  goto_programt::targett tmp_head = loop_head;
  for (; tmp_head != loop_exit; tmp_head++)
  {
    auto it = cache.find(tmp_head->location_number);
    if (it != cache.end())
    {
      // Re-inject the guards the first visit collected here. Storing
      // only `reaches` lost these guards on every cache hit and silently
      // weakened the entry-condition assume.
      guards.insert(
        it->second.guards_to_merge.begin(), it->second.guards_to_merge.end());
      return it->second.reaches;
    }

    /* TODO: disable this for now, it will be used for termination evaluation
     * in the future.

    // Return, assume(0) and assert(0) stop the execution, so ignore these
    // branches too
    if(tmp_head->is_return())
      return true;

    if(tmp_head->is_assume() || tmp_head->is_assert())
      if(is_false(tmp_head->guard))
        return true;
    */

    if (tmp_head->is_goto() && !tmp_head->is_backwards_goto())
    {
      expr2tc g = tmp_head->guard;
      simplify(g);

      // If the guard is false, we can skip it right away
      if (is_false(g))
        continue;

      // We need to walk the branches and collect constraints that force
      // the path inside the loop and reach the end of the loop body

      // Get the branch number for caching
      auto const branch_number = tmp_head->location_number;

      // Walk the true branch
      bool true_branch = true;
      guardst true_branch_guard;
      if (!is_false(g))
      {
        true_branch_guard[branch_number].add(g);
        true_branch = get_entry_cond_rec(
          tmp_head->targets.front(), loop_exit, true_branch_guard, cache);
      }

      // Walk the false branch
      bool false_branch = true;
      guardst false_branch_guard;
      if (!is_true(g))
      {
        goto_programt::targett new_tmp_head = tmp_head;
        make_not(g);
        false_branch_guard[branch_number].add(g);
        false_branch = get_entry_cond_rec(
          ++new_tmp_head, loop_exit, false_branch_guard, cache);
      }

      // Cache: store BOTH the recursion's reach-status at this branch
      // AND the guards that should be re-injected on a later cache hit.
      branch_cache_entryt entry;
      entry.reaches = false_branch && true_branch;

      // If both sides reach the end of the loop or if neither reaches it
      // we can ignore them
      if (!(false_branch ^ true_branch))
      {
        cache[branch_number] = std::move(entry);
        return false_branch && true_branch;
      }

      // At least only one of the branches reach the end of the loop, so
      // collect the guards from the non-reaching side.
      if (!true_branch)
      {
        guards.insert(true_branch_guard.begin(), true_branch_guard.end());
        entry.guards_to_merge = std::move(true_branch_guard);
        cache[branch_number] = std::move(entry);
        return false;
      }

      if (!false_branch)
      {
        guards.insert(false_branch_guard.begin(), false_branch_guard.end());
        entry.guards_to_merge = std::move(false_branch_guard);
        cache[branch_number] = std::move(entry);
        return false;
      }
    }
  }

  return false;
}

void make_nondet_assign(
  goto_functiont &goto_function,
  goto_programt::targett &loop_head,
  const loopst &loop)
{
  // Track the original loop head
  auto const original_loop_head = loop_head;

  // Check if the loop_head is an assertion, and track it
  const bool is_assert = loop_head->is_assert();

  // If it's an assertion, adjust loop_head to insert assignments before it
  if ((is_assert) && loop_head != goto_function.body.instructions.begin())
  {
    --loop_head;
    // We add instructions before a GOTO instruction
    // So we ensure we have one here
    assert(loop_head->is_goto());
  }

  // Get the list of variables modified inside the loop
  auto const &loop_vars = loop.get_modified_loop_vars();

  goto_programt dest;
  size_t inserted = 0;
  for (auto const &lhs : loop_vars)
  {
    // Generate a nondeterministic value for the loop variable
    expr2tc rhs = gen_nondet(lhs->type);

    // Create an assignment instruction for the nondeterministic value
    goto_programt::targett t = dest.add_instruction(ASSIGN);
    t->inductive_step_instruction = true;
    t->code = code_assign2tc(lhs, rhs);
    // Keep the same location as the loop head
    t->location = loop_head->location;
    ++inserted;
  }

  // Insert the generated assignments before the loop head in the program
  goto_function.body.insert_swap(loop_head, dest);

  // Restore loop_head to its original position. insert_swap leaves
  // loop_head pointing at the *first* inserted instruction (or, when
  // nothing was inserted, at the original loop head). We must advance
  // it by exactly `inserted` positions so it ends up back at the
  // original loop_head — and never further, even if a prior pass
  // (e.g. assume_loop_entry_cond_before_loop) already placed an
  // inductive_step_instruction right after the loop_head: the old
  // "walk while inductive_step_instruction" heuristic happily swallowed
  // that ASSUME too, leaving the back-edge retargeted past the loop's
  // exit IF and the loop guard bypassed in BC/FC.
  if (is_assert)
  {
    // Restore the original loop head if it was an assertion
    loop_head = original_loop_head;
    assert(loop_head->is_assert());
  }
  else
  {
    for (size_t i = 0; i < inserted; ++i)
      ++loop_head;
  }
}

bool contains_rec(const expr2tc &expr, const loopst::loop_varst &vars)
{
  // Check this node first: if it's a tracked symbol, we're done.
  if (is_symbol2t(expr) && vars.find(expr) != vars.end())
    return true;

  // Otherwise recurse into operands and stop at the first match.
  bool res = false;
  expr->foreach_operand([&vars, &res](const expr2tc &e) {
    if (res || is_nil_expr(e))
      return;
    res = contains_rec(e, vars);
  });
  return res;
}

void remove_unrelated_loop_cond(guardst &guards, const loopst &loop)
{
  auto const &loop_vars = loop.get_modified_loop_vars();
  if (!loop_vars.size())
  {
    guards.clear();
    return;
  }

  guardst::iterator g = guards.begin();
  while (g != guards.end())
  {
    expr2tc g_expr = g->second.as_expr();

    if (!contains_rec(g_expr, loop_vars))
      g = guards.erase(g);
    else
      ++g;
  }
}

void assume_loop_entry_cond_before_loop(
  goto_functiont &goto_function,
  goto_programt::targett &loop_head,
  const guardst &guards)
{
  // Combine all per-branch loop-entry guards into one ASSUME and
  // insert it via raw `insert(loop_head, ...)` immediately before the
  // loop_head IF. This runs *after* make_nondet_assign in
  // transform_loop, so the layout becomes
  //
  //   NONDET havocs               [inductive_step]   <- make_nondet_assign
  //   ASSUME(entry_cond)          [inductive_step]   <- this insert
  //   loop_head: IF !cond GOTO exit                  <- original IF
  //   body
  //   GOTO loop_head  -> back-edge targets the IF
  //   exit:
  //
  // Two properties matter for soundness/precision:
  //   1. The ASSUME comes *after* the NONDET havocs, so it constrains
  //      the just-havoced state (the actual inductive-hypothesis pin),
  //      not the pre-loop concrete values.
  //   2. The ASSUME sits *before* loop_head and is reached only via
  //      fall-through from the havocs. The back-edge
  //      (adjust_loop_head_and_exit retargets it at loop_head, the
  //      IF) skips both the havocs and the ASSUME on every iteration
  //      after the first, so the natural exit on `cond` is never
  //      blocked by a re-firing entry-cond assume.
  //
  // The legacy `insert_swap(tmp_head, ASSUME)` placement (one ASSUME
  // per branch, swapped *at* the branch instruction) was unsound
  // for the loop_head IF because insert_swap pins external jumps to
  // the iterator: back-edges then landed on the ASSUME and re-fired
  // it on every iteration, killing the natural-exit path in IS and
  // producing vacuous UNSAT proofs (e.g. SV-COMP
  // sll_of_sll_nondet_append-2). The earlier fix that combined the
  // ASSUME and placed it *after* loop_head fixed SLL but over-
  // constrained loops where the body modifies a loop-exit variable
  // (e.g. `i++` in array_3-1): the back-edge re-evaluated the
  // ASSUME on the iteration where the body's increment had pushed
  // the variable past the exit condition, again killing a natural
  // path.
  //
  // Iterate the collected guards directly instead of walking the
  // instruction range and looking each one up by location_number.
  // After make_nondet_assign's insert_swap, location_number does NOT
  // travel with the instruction content (instructiont::swap exchanges
  // code, type, guard, targets, ... but not location_number — see
  // goto_program.h), so the iterator advance leaves loop_head pointing
  // at the original IF carrying the freshly-inserted slot's default
  // location_number=0 instead of the IF's original number. The
  // walk-and-lookup approach then misses every guard, the combined
  // entry condition comes out empty, and no ASSUME is inserted before
  // the IF — losing the inductive-hypothesis pin and regressing
  // post-loop assertions in loop-invariants, incremental-smt,
  // witnesses_validate, and esbmc-solidity tests (issue #4846).
  // Conjunction is commutative, so iteration order is irrelevant.
  guard2tc combined;
  for (auto const &kv : guards)
  {
    expr2tc loop_cond = kv.second.as_expr();

    if (is_nil_expr(loop_cond) || is_true(loop_cond))
      continue;

    // A guard that simplifies to false would make the assume kill the
    // path even before the loop. Preserve the legacy "bail out" choice:
    // any unresolved false among the branch guards skips the whole
    // entry-cond instrumentation.
    if (is_false(loop_cond))
      return;

    combined.add(loop_cond);
  }

  expr2tc combined_expr = combined.as_expr();
  if (
    is_nil_expr(combined_expr) || is_true(combined_expr) ||
    is_false(combined_expr))
    return;

  goto_programt::instructiont instruction;
  instruction.type = ASSUME;
  instruction.guard = combined_expr;
  instruction.inductive_step_instruction = true;
  instruction.location = loop_head->location;
  goto_function.body.instructions.insert(loop_head, instruction);
}

void adjust_loop_head_and_exit(
  goto_programt::targett &loop_head,
  goto_programt::targett &loop_exit)
{
  loop_exit->targets.clear();
  loop_exit->targets.push_front(loop_head);

  goto_programt::targett _loop_exit = loop_exit;
  ++_loop_exit;

  // Zero means that the instruction was added during
  // the k-induction transformation
  if (_loop_exit->location_number == 0)
  {
    // Clear the target
    loop_head->targets.clear();

    // And set the target to be the newly inserted assume(cond)
    loop_head->targets.push_front(_loop_exit);
  }
}

/// Per-loop k-induction transformation: havoc each loop's modified
/// variables and inject an ASSUME of the loop entry condition right
/// before the loop head.
void transform_loop(goto_functiont &goto_function, loopst &loop)
{
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();

  // Loop-scoped cache for get_entry_cond_rec. The cache key is the
  // IF's location_number; nested loops in the same function don't
  // reuse entries because we construct a fresh cache per call.
  marked_branchst cache;
  guardst guards;
  get_entry_cond_rec(loop_head, loop_exit, guards, cache);

  // Remove loop conditions not related to the written variables
  remove_unrelated_loop_cond(guards, loop);

  // Order matters: the entry-cond ASSUME must constrain the state
  // *after* the loop vars have been havoced, so we emit the NONDET
  // havocs first and then place the ASSUME just before loop_head.
  // make_nondet_assign also rewinds loop_head back onto the original
  // loop head (post its advance-by-`inserted` step), so by the time
  // assume_loop_entry_cond_before_loop runs the iterator is correct
  // and the ASSUME ends up between the havocs and the IF.

  // Create the nondet assignments on the beginning of the loop
  make_nondet_assign(goto_function, loop_head, loop);

  // Assume the loop entry condition before going into the loop
  assume_loop_entry_cond_before_loop(goto_function, loop_head, guards);

  // Check if the loop exit needs to be updated. We must point to the
  // assume that was inserted in the previous transformation
  adjust_loop_head_and_exit(loop_head, loop_exit);
}

/// Phase 2 (#5230): for a loop that writes array elements through a pointer,
/// resolve every directly-written pointer against the value-set fixpoint @p
/// vsa and inject the referenced named objects into the loop's modified-
/// variable set, so make_nondet_assign havocs them as whole symbols. The
/// fixpoint is a sound over-approximation of the objects each pointer may
/// reference at the loop head, so havocing all of them generalises the loop
/// at least as much as its real effect — the inductive hypothesis is no
/// longer too strong.
///
/// Returns true iff every write resolved to concrete named objects (the
/// inductive step can stay enabled); returns false to abstain, in which case
/// the caller disables the inductive step (the conservative Phase 1
/// behaviour). We abstain whenever the points-to set is empty, unknown,
/// invalid, or contains a heap (dynamic) object — none of which has a
/// nameable symbol to havoc.
bool resolve_pointer_array_writes(loopst &loop, value_set_analysist &vsa)
{
  // A write reaching the loop through a callee (pointer is a callee
  // parameter, not in scope here) or with no extractable pointer cannot be
  // resolved at the loop head.
  if (loop.pointer_array_write_unresolvable())
    return false;

  const auto &ptrs = loop.get_pointer_array_write_ptrs();
  if (ptrs.empty())
    return false;

  goto_programt::const_targett loc = loop.get_original_loop_head();
  if (!vsa.has_location(loc))
    return false;

  loopst::loop_varst objects;
  for (const expr2tc &ptr : ptrs)
  {
    value_setst::valuest values;
    vsa.get_values(loc, ptr, values);

    // No points-to information means the pointer could reference anything.
    if (values.empty())
      return false;

    for (const expr2tc &v : values)
    {
      // Only a concrete, named object can be havoc'd as a whole symbol.
      // unknown / invalid / heap (dynamic) objects have no nameable symbol,
      // so abstain and let Phase 1 disable the inductive step.
      if (!is_object_descriptor2t(v))
        return false;
      const expr2tc &object = to_object_descriptor2t(v).object;
      if (!is_symbol2t(object) || !check_var_name(object))
        return false;
      objects.insert(object);
    }
  }

  // Every write resolved: havoc each referenced object as a whole symbol.
  for (const expr2tc &obj : objects)
    loop.add_modified_var_to_loop(obj);
  return true;
}

/// True iff any user function directly writes an array element through a
/// pointer (`is_assign` whose LHS `indexes_through_pointer`). Used to decide
/// whether to build the value-set fixpoint at all. Non-mutating, so it is
/// safe to run before goto_loopst construction (which can rewrite self-loops)
/// and before any transform_loop. Over-approximates: a write outside any loop
/// also counts, which only ever builds the fixpoint unnecessarily (never
/// wrong). The via-callee write path always abstains without the fixpoint, so
/// it need not be detected here. See #5230.
bool has_direct_pointer_array_write(const goto_functionst &goto_functions)
{
  forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available || it->second.body.hide)
      continue;
    for (const auto &instr : it->second.body.instructions)
    {
      if (!instr.is_assign())
        continue;
      const expr2tc &target = to_code_assign2t(instr.code).target;
      // Only array-element writes through a pointer are unsound for the
      // inductive step (the pointee array has no nameable symbol). Struct
      // member writes through a pointer are fine — the struct symbol can be
      // havoc'd as a whole. Mirror the is_index2t guard in
      // collect_lhs_symbols/modifies_pointer_array. See #5230.
      if (is_index2t(target) &&
          indexes_through_pointer(to_index2t(target).source_value))
        return true;
    }
  }
  return false;
}
} // namespace

bool goto_k_induction(goto_functionst &goto_functions, const namespacet &ns)
{
  // Build the value-set fixpoint once, up front, on the pristine program —
  // value_set_analysist asserts on k-induction-transformed CFGs, so it must
  // run before any transform_loop mutation, not lazily from inside the loop
  // (an earlier plain loop would already have been transformed). Built only
  // when a pointer-array write is present, to avoid paying the cost.
  std::shared_ptr<value_set_analysist> vsa;
  if (has_direct_pointer_array_write(goto_functions))
  {
    vsa = std::make_shared<value_set_analysist>(ns);
    try
    {
      (*vsa)(goto_functions);
    }
    catch (...)
    {
      // VSA is best-effort: any failure (incomplete implementation, symbolic
      // type, ...) just means we cannot resolve pointees and fall back to
      // disabling the inductive step.
      vsa = nullptr;
    }
  }

  bool disable_inductive_step = false;
  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available)
      continue;
    // Library helpers (body.hide) write through pointers in nearly every
    // model (memcpy, string ops, ...). Letting them set the gate would
    // disable the inductive step for essentially every program. Their loops
    // are never the user property's witness, so exclude them from the
    // decision — mirrors the body.hide guard in goto_termination.
    const bool user_function = !it->second.body.hide;
    goto_loopst loops(it->first, goto_functions, it->second);
    for (auto &loop : loops.get_loops())
    {
      // A loop that writes an array element through a pointer cannot be
      // havoc'd as a named symbol by the inductive step. Phase 2 tries to
      // resolve the written pointer to concrete named objects and havoc
      // those instead, keeping the inductive step sound and enabled. If the
      // pointee cannot be resolved, fall back to Phase 1: disable the
      // inductive step. Checked before the empty-modified-set skip below,
      // since such a loop may have no named modified vars yet. See #5230.
      if (
        user_function && loop.modifies_pointer_array() &&
        !(vsa && resolve_pointer_array_writes(loop, *vsa)))
        disable_inductive_step = true;

      if (loop.get_modified_loop_vars().empty())
        continue;
      transform_loop(it->second, loop);
    }
  }
  goto_functions.update();
  return disable_inductive_step;
}
