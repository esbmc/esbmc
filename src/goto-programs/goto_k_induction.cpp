/*
 * goto_unwind.cpp
 *
 *  Created on: Jun 3, 2015
 *      Author: mramalho
 */

#include <goto-programs/goto_k_induction.h>
#include <goto-programs/remove_skip.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/std_expr.h>

void goto_k_induction(
  goto_functionst &goto_functions,
  message_handlert &message_handler)
{
  Forall_goto_functions(it, goto_functions)
    if(it->second.body_available)
      goto_k_inductiont(it->first, goto_functions, it->second, message_handler);

  goto_functions.update();
}

void goto_k_inductiont::goto_k_induction()
{
  // Full unwind the program
  for(auto &function_loop : function_loops)
  {
    if(function_loop.get_loop_vars().empty())
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

  auto loop_termination_cond = get_termination_cond(loop_head, loop_exit);

  // Assume the loop entry condition before go into the loop
  assume_loop_entry_cond_before_loop(loop_head);

  // Create the nondet assignments on the beginning of the loop
  make_nondet_assign(loop_head, loop);

  // Check if the loop exit needs to be updated
  // We must point to the assume that was inserted in the previous
  // transformation
  adjust_loop_head_and_exit(loop_head, loop_exit);

  assume_neg_loop_cond_after_loop(loop_exit, loop_termination_cond);
}

const expr2tc goto_k_inductiont::get_termination_cond(
  goto_programt::targett &loop_head,
  goto_programt::targett &loop_exit)
{
  // Let's not change the loop_head or loop_exit

  goto_programt::targett tmp_exit = loop_exit;

  // Get the loop exit number
  auto const &exit_number = (++tmp_exit)->location_number;

  // Collect the loop termination conditions
  expr2tc cond = gen_false_expr();

  for(goto_programt::targett tmp_head = loop_head; tmp_head != loop_exit;
      tmp_head++)
  {
    // If there is a jump to outside the loop, collect it
    if(
      tmp_head->is_goto() &&
      (tmp_head->targets.front()->location_number >= exit_number))
    {
      cond = or2tc(cond, tmp_head->guard);
    }
  }

  // If we hit the loop's end and didn't find any loop condition
  // return a nil exprt
  if(cond == gen_false_expr())
    return expr2tc();

  // Remove first or (for some reason, simplification is not working)
  cond = to_or2t(cond).side_2;

  // We found the loop condition however it's inverted, so we
  // have to negate it before returning
  make_not(cond);
  return cond;
}

void goto_k_inductiont::make_nondet_assign(
  goto_programt::targett &loop_head,
  const loopst &loop)
{
  auto const &loop_vars = loop.get_loop_vars();

  goto_programt dest;
  for(auto const &lhs : loop_vars)
  {
    expr2tc rhs = sideeffect2tc(
      lhs->type,
      expr2tc(),
      expr2tc(),
      std::vector<expr2tc>(),
      type2tc(),
      sideeffect2t::nondet);

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

void goto_k_inductiont::assume_loop_entry_cond_before_loop(
  goto_programt::targett &loop_head)
{
  auto loop_cond = loop_head->guard;

  // It might be nil, in case of goto loops
  if(is_nil_expr(loop_cond))
    return;

  // Otherwise, the loop entry condition will be negated, so revert it
  make_not(loop_cond);

  if(is_true(loop_cond))
    return;

  goto_programt dest;
  assume_cond(loop_cond, dest, loop_head->location);

  goto_function.body.insert_swap(loop_head, dest);
}

void goto_k_inductiont::assume_neg_loop_cond_after_loop(
  goto_programt::targett &loop_exit,
  expr2tc &loop_cond)
{
  goto_programt dest;
  expr2tc neg_loop_cond = loop_cond;
  make_not(neg_loop_cond);
  assume_cond(neg_loop_cond, dest, loop_exit->location);

  goto_function.body.insert_swap(++loop_exit, dest);
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
