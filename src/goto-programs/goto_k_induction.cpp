#include <goto-programs/goto_k_induction.h>
#include <goto-programs/remove_skip.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/std_expr.h>

void goto_k_induction(goto_functionst &goto_functions)
{
  Forall_goto_functions(it, goto_functions)
    if(it->second.body_available)
      goto_k_inductiont(it->first, goto_functions, it->second);

  goto_functions.update();
}

void goto_termination(goto_functionst &goto_functions)
{
  Forall_goto_functions(it, goto_functions)
    if(it->second.body_available)
      goto_k_inductiont(it->first, goto_functions, it->second);
  goto_functions.update();

  auto function = goto_functions.function_map.find("__ESBMC_main");

  // Search for __ESBMC_main
  auto it = function->second.body.instructions.begin();
  while(it != function->second.body.instructions.end())
  {
    if(it->is_function_call())
    {
      auto const &call = to_code_function_call2t(it->code);
      if(to_symbol2t(call.function).thename.as_string() == "c:@F@main")
        break;
    }
    it++;
  }
  assert(it != function->second.body.instructions.end());

  // Create an assert(0)
  goto_programt dest;
  goto_programt::targett t = dest.add_instruction(ASSERT);
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
  for(auto &function_loop : function_loops)
  {
    if(function_loop.get_modified_loop_vars().empty())
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
  if(entry_number > exit_number)
    return true;

  goto_programt::targett tmp_head = loop_head;
  for(; tmp_head != loop_exit; tmp_head++)
  {
    auto it = marked_branch.find(tmp_head->location_number);
    if(it != marked_branch.end())
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

    if(tmp_head->is_goto() && !tmp_head->is_backwards_goto())
    {
      expr2tc g = tmp_head->guard;
      simplify(g);

      // If the guard is false, we can skip it right away
      if(is_false(g))
        continue;

      // We need to walk the branches and collect constraints that force
      // the path inside the loop and reach the end of the loop body

      // Get the branch number for caching
      auto const branch_number = tmp_head->location_number;

      // Walk the true branch
      bool true_branch = true;
      guardst true_branch_guard;
      if(!is_false(g))
      {
        true_branch_guard[branch_number].add(g);
        true_branch = get_entry_cond_rec(
          tmp_head->targets.front(), loop_exit, true_branch_guard);
      }

      // Walk the false branch
      bool false_branch = true;
      guardst false_branch_guard;
      if(!is_true(g))
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
      if(!(false_branch ^ true_branch))
        return false_branch && true_branch;

      // At least only one of the branches reach the end of the loop, so
      // collect the guards
      if(!true_branch)
      {
        guards.insert(true_branch_guard.begin(), true_branch_guard.end());
        return false;
      }

      if(!false_branch)
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
  auto const &loop_vars = loop.get_modified_loop_vars();

  goto_programt dest;
  for(auto const &lhs : loop_vars)
  {
    // do not assign nondeterministic value to pointers if we assume
    // objects extracted from the value set analysis
    if(
      !config.options.get_bool_option("no-pointer-check") &&
      config.options.get_bool_option("add-symex-value-sets") &&
      is_pointer_type(lhs))
      continue;
    expr2tc rhs = gen_nondet(lhs->type);

    goto_programt::targett t = dest.add_instruction(ASSIGN);
    t->inductive_step_instruction = true;
    t->code = code_assign2tc(lhs, rhs);
    t->location = loop_head->location;
  }

  goto_function.body.insert_swap(loop_head, dest);

  // Get original head again
  // Since we are using insert_swap to keep the targets, the
  // original loop head as shifted to after the assume cond
  while((++loop_head)->inductive_step_instruction)
    ;
}

static bool contains_rec(const expr2tc &expr, const loopst::loop_varst &vars)
{
  bool res = false;
  expr->foreach_operand([&vars, &res](const expr2tc &e) {
    if(!is_nil_expr(e))
      res = contains_rec(e, vars) || res;
    return res;
  });

  if(!is_symbol2t(expr))
    return res;

  return (vars.find(expr) != vars.end()) || res;
}

void goto_k_inductiont::remove_unrelated_loop_cond(
  guardst &guards,
  loopst &loop)
{
  auto const &loop_vars = loop.get_modified_loop_vars();
  if(!loop_vars.size())
  {
    guards.clear();
    return;
  }

  guardst::iterator g = guards.begin();
  while(g != guards.end())
  {
    expr2tc g_expr = g->second.as_expr();

    if(!contains_rec(g_expr, loop_vars))
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
  for(; tmp_head != loop_exit; tmp_head++)
  {
    auto const g = guards.find(tmp_head->location_number);
    if(g == guards.end())
      continue;

    expr2tc loop_cond = g->second.as_expr();

    if(is_nil_expr(loop_cond))
      return;

    if(is_true(loop_cond) || is_false(loop_cond))
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
  if(_loop_exit->location_number == 0)
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
