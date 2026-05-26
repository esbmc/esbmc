#include <goto-programs/goto_k_induction.h>
#include <goto-programs/remove_no_op.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/std_expr.h>

void goto_k_induction(goto_functionst &goto_functions)
{
  Forall_goto_functions (it, goto_functions)
    if (it->second.body_available)
      goto_k_inductiont(it->first, goto_functions, it->second);

  goto_functions.update();
}

void goto_termination(goto_functionst &goto_functions)
{
  Forall_goto_functions (it, goto_functions)
    if (it->second.body_available)
      goto_k_inductiont(it->first, goto_functions, it->second);
  goto_functions.update();

  auto function = goto_functions.function_map.find("__ESBMC_main");

  // Search for __ESBMC_main
  auto it = function->second.body.instructions.begin();
  while (it != function->second.body.instructions.end())
  {
    if (it->is_function_call())
    {
      auto const &call = to_code_function_call2t(it->code);
      if (to_symbol2t(call.function).thename.as_string() == "c:@F@main")
        break;
    }
    it++;
  }
  assert(it != function->second.body.instructions.end());

  // Create assert(0) as termination marker.
  // This assertion fails when reached, allowing reachability analysis
  // to detect program termination vs. infinite execution
  goto_programt dest;
  goto_programt::targett t = dest.add_instruction(ASSERT);
  // Always false - assertion always fails when reached
  t->guard = gen_false_expr();
  t->inductive_step_instruction = true;
  t->inductive_assertion = false;
  t->location.comment("termination");

  // And add it one instruction after the call to main
  it++;
  function->second.body.insert_swap(it, dest);
}

void goto_k_inductiont::goto_k_induction()
{
  // Full unwind the program
  for (auto &function_loop : function_loops)
  {
    if (function_loop.get_modified_loop_vars().empty())
      continue;

    // Start the loop conversion
    convert_finite_loop(function_loop);
  }
}

void goto_k_inductiont::convert_finite_loop(loopst &loop)
{
  // Get current loop head and loop exit
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();

  // The branch cache is loop-scoped: a cached `reaches` value is computed
  // against this loop's `loop_exit`, and the cached guards were collected
  // along this loop's recursive walk. Nested loops share the goto_k_inductiont
  // instance (one per function), so without clearing here the outer loop's
  // entries would be reused by the inner loop with a different `loop_exit`.
  marked_branch.clear();

  guardst guards;
  get_entry_cond_rec(loop_head, loop_exit, guards);

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
  make_nondet_assign(loop_head, loop);

  // Assume the loop entry condition before going into the loop
  assume_loop_entry_cond_before_loop(loop_head, guards);

  // Check if the loop exit needs to be updated
  // We must point to the assume that was inserted in the previous
  // transformation
  adjust_loop_head_and_exit(loop_head, loop_exit);
}

bool goto_k_inductiont::get_entry_cond_rec(
  const goto_programt::targett &loop_head,
  const goto_programt::targett &loop_exit,
  guardst &guards)
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
    auto it = marked_branch.find(tmp_head->location_number);
    if (it != marked_branch.end())
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
          tmp_head->targets.front(), loop_exit, true_branch_guard);
      }

      // Walk the false branch
      bool false_branch = true;
      guardst false_branch_guard;
      if (!is_true(g))
      {
        goto_programt::targett new_tmp_head = tmp_head;
        make_not(g);
        false_branch_guard[branch_number].add(g);
        false_branch =
          get_entry_cond_rec(++new_tmp_head, loop_exit, false_branch_guard);
      }

      // Cache: store BOTH the recursion's reach-status at this branch
      // AND the guards that should be re-injected on a later cache hit.
      branch_cache_entryt entry;
      entry.reaches = false_branch && true_branch;

      // If both sides reach the end of the loop or if neither reaches it
      // we can ignore them
      if (!(false_branch ^ true_branch))
      {
        marked_branch[branch_number] = std::move(entry);
        return false_branch && true_branch;
      }

      // At least only one of the branches reach the end of the loop, so
      // collect the guards from the non-reaching side.
      if (!true_branch)
      {
        guards.insert(true_branch_guard.begin(), true_branch_guard.end());
        entry.guards_to_merge = std::move(true_branch_guard);
        marked_branch[branch_number] = std::move(entry);
        return false;
      }

      if (!false_branch)
      {
        guards.insert(false_branch_guard.begin(), false_branch_guard.end());
        entry.guards_to_merge = std::move(false_branch_guard);
        marked_branch[branch_number] = std::move(entry);
        return false;
      }
    }
  }

  return false;
}

void goto_k_inductiont::make_nondet_assign(
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
  const bool skip_pointer_havoc =
    config.options.get_bool_option("add-symex-value-sets");
  for (auto const &lhs : loop_vars)
  {
    // Under --add-symex-value-sets, leave pointer LHSs untouched: the
    // value-set analysis pins them to the over-approximated object set
    // via assume() in symex_dereference, and a nondet havoc here would
    // override that pin with a fresh, unconstrained pointer — making
    // the inductive step's post-loop reasoning lose track of which
    // objects the pointer can alias (e.g. regression/incremental-smt/
    // incremental-{18,24}, which rely on r1,r2 ∈ {&x,&y} surviving
    // havoc, issue #4846).
    if (skip_pointer_havoc && is_pointer_type(lhs))
      continue;

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

static bool contains_rec(const expr2tc &expr, const loopst::loop_varst &vars)
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

void goto_k_inductiont::remove_unrelated_loop_cond(
  guardst &guards,
  loopst &loop)
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

void goto_k_inductiont::assume_loop_entry_cond_before_loop(
  goto_programt::targett &loop_head,
  const guardst &guards)
{
  // Combine all per-branch loop-entry guards into one ASSUME and
  // insert it via raw `insert(loop_head, ...)` immediately before the
  // loop_head IF. This runs *after* make_nondet_assign in
  // convert_finite_loop, so the layout becomes
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

void goto_k_inductiont::adjust_loop_head_and_exit(
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

void goto_k_inductiont::assume_cond(
  const expr2tc &cond,
  goto_programt &dest,
  const locationt &loc)
{
  goto_programt tmp_e;
  goto_programt::targett e = tmp_e.add_instruction(ASSUME);
  e->inductive_step_instruction = true;
  e->guard = cond;
  e->location = loc;
  dest.destructive_append(tmp_e);
}
