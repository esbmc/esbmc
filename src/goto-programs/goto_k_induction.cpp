#include <goto-programs/goto_k_induction.h>
#include <goto-programs/remove_no_op.h>
#include <util/c_types.h>
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
  // Termination as a safety property. Two transformations:
  //
  //   1. Apply the k-induction loop transformation (havoc loop vars +
  //      assume entry condition) to every loop. The inductive step
  //      then runs from an arbitrary loop state — so if it cannot
  //      reach end-of-main, no iterate of the loop can.
  //   2. Insert `assert(false)` right after the call to `main()` in
  //      `__ESBMC_main`. The reduction is:
  //
  //        program does NOT terminate  iff  assert is unreachable
  //
  //      In the inductive step, UNSAT (assert unreachable from havoc'd
  //      state) proves non-termination outright. In the base case,
  //      SAT (assert reachable from concrete initial state) refutes it.
  // Apply k-induction (havoc + assume_entry) to every function except
  // __ESBMC_main: that's where we insert the termination marker, and
  // running k-induction there would scan its (loop-free) body and
  // potentially mutate it.
  Forall_goto_functions (it, goto_functions)
    if (it->second.body_available && it->first != "__ESBMC_main")
      goto_k_inductiont(it->first, goto_functions, it->second);
  goto_functions.update();

  auto function = goto_functions.function_map.find("__ESBMC_main");
  if (function == goto_functions.function_map.end())
    return;

  // Place the termination marker right before __ESBMC_main's
  // END_FUNCTION. Inserting after a specific FUNCTION_CALL (e.g. the
  // c:@F@main call) would be fragile — C++ name mangling, the Python
  // frontend (python_user_main), and --function-renamed entries all
  // produce different callee symbols, and assert-aborting on a
  // missing match would crash the strategy. Inserting at end-of-
  // wrapper is shape-agnostic: control reaches it exactly when the
  // program is about to return from __ESBMC_main, which is the
  // semantic point we want to detect.
  auto end_it = function->second.body.instructions.begin();
  while (end_it != function->second.body.instructions.end() &&
         end_it->type != END_FUNCTION)
    ++end_it;
  if (end_it == function->second.body.instructions.end())
    return; // no END_FUNCTION; nothing safe to anchor to

  // Insert `assert(false)` immediately before __ESBMC_main's
  // END_FUNCTION. Mark it `inductive_step_instruction` so that
  // execution_state.cpp:215 skips it during base_case and
  // forward_condition (we don't want FC's "all states reachable"
  // proof to be derailed by an always-false VCC) — it only fires in
  // the inductive step, where it serves as the "did we reach end-of-
  // main from a havoc'd iterate?" probe.
  //
  // Use `insert` (not `insert_swap`): we want to INSERT before the
  // target, preserving it. insert_swap would overwrite the target.
  auto inserted = function->second.body.instructions.insert(
    end_it, goto_programt::instructiont());
  inserted->type = ASSERT;
  inserted->guard = gen_false_expr();
  inserted->location.comment("termination");
  inserted->function = "__ESBMC_main";
  inserted->inductive_step_instruction = true;

  goto_functions.update();
}

bool has_pointer_only_loop(goto_functionst &goto_functions)
{
  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available || it->first == "__ESBMC_main")
      continue;
    goto_loopst loops(it->first, goto_functions, it->second);
    for (auto &loop : loops.get_loops())
    {
      // Exclude empty modified sets — those are skipped by
      // goto_k_induction's first check (line 88) regardless of
      // --add-symex-value-sets, and contains_only_pointers vacuously
      // returns true on an empty set. The IS-soundness gate we want
      // here is specifically the value-sets + pointer-only combo
      // (the line 91-94 skip).
      if (loop.get_modified_loop_vars().empty())
        continue;
      if (loop.contains_only_pointers())
        return true;
    }
  }
  return false;
}

void goto_k_inductiont::goto_k_induction()
{
  // Full unwind the program
  for (auto &function_loop : function_loops)
  {
    if (function_loop.get_modified_loop_vars().empty())
      continue;

    if (
      config.options.get_bool_option("add-symex-value-sets") &&
      function_loop.contains_only_pointers())
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

  guardst guards;
  get_entry_cond_rec(loop_head, loop_exit, guards);

  // Remove loop conditions not related to the written variables
  remove_unrelated_loop_cond(guards, loop);

  // Assume the loop entry condition before go into the loop
  assume_loop_entry_cond_before_loop(loop_head, loop_exit, guards);

  // Create the nondet assignments on the beginning of the loop
  make_nondet_assign(loop_head, loop);

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
      return it->second;

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

      // If we evaluated both sides of the branch, mark it so we don't
      // have to do it again.
      marked_branch[branch_number] = (false_branch ^ true_branch);

      // If both side reach the end of the loop or if both side don't reach it
      // we can ignore them
      if (!(false_branch ^ true_branch))
        return false_branch && true_branch;

      // At least only one of the branches reach the end of the loop, so
      // collect the guards
      if (!true_branch)
      {
        guards.insert(true_branch_guard.begin(), true_branch_guard.end());
        return false;
      }

      if (!false_branch)
      {
        guards.insert(false_branch_guard.begin(), false_branch_guard.end());
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
  for (auto const &lhs : loop_vars)
  {
    // do not assign nondeterministic value to pointers if we assume
    // objects extracted from the value set analysis
    if (
      config.options.get_bool_option("add-symex-value-sets") &&
      is_pointer_type(lhs))
      continue;

    // Generate a nondeterministic value for the loop variable
    expr2tc rhs = gen_nondet(lhs->type);

    // Create an assignment instruction for the nondeterministic value
    goto_programt::targett t = dest.add_instruction(ASSIGN);
    t->inductive_step_instruction = true;
    t->code = code_assign2tc(lhs, rhs);
    // Keep the same location as the loop head
    t->location = loop_head->location;
  }

  // Insert the generated assignments before the loop head in the program
  goto_function.body.insert_swap(loop_head, dest);

  // Get original head again
  // Since we are using insert_swap to keep the targets, the
  // original loop head as shifted to after the assume cond
  if (is_assert)
  {
    // Restore the original loop head if it was an assertion
    loop_head = original_loop_head;
    assert(loop_head->is_assert());
  }
  else
  {
    // Move past the inserted instructions during the inductive step
    while ((++loop_head)->inductive_step_instruction)
      ;
  }
}

static bool contains_rec(const expr2tc &expr, const loopst::loop_varst &vars)
{
  bool res = false;
  expr->foreach_operand([&vars, &res](const expr2tc &e) {
    if (!is_nil_expr(e))
      res = contains_rec(e, vars) || res;
    return res;
  });

  if (!is_symbol2t(expr))
    return res;

  return (vars.find(expr) != vars.end()) || res;
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
  goto_programt::targett &loop_exit,
  const guardst &guards)
{
  goto_programt::targett tmp_head = loop_head;
  for (; tmp_head != loop_exit; tmp_head++)
  {
    auto const g = guards.find(tmp_head->location_number);
    if (g == guards.end())
      continue;

    expr2tc loop_cond = g->second.as_expr();

    if (is_nil_expr(loop_cond))
      return;

    if (is_true(loop_cond) || is_false(loop_cond))
      return;

    goto_programt dest;
    assume_cond(loop_cond, dest, tmp_head->location);

    goto_function.body.insert_swap(tmp_head, dest);
  }
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
