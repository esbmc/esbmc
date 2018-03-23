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
  contextt &context,
  message_handlert &message_handler)
{
  Forall_goto_functions(it, goto_functions)
    if(it->second.body_available)
      goto_k_inductiont(
        it->first, goto_functions, it->second, context, message_handler);

  goto_functions.update();
}

void goto_k_inductiont::goto_k_induction()
{
  // Full unwind the program
  for(auto &function_loop : function_loops)
  {
    if(function_loop.get_loop_vars().empty())
      continue;

    // TODO: Can we check if the loop is infinite? If so, we should
    // disable the forward condition

    // Start the loop conversion
    convert_finite_loop(function_loop);
  }
}

void goto_k_inductiont::convert_finite_loop(loopst &loop)
{
  // Get current loop head and loop exit
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();

  auto loop_cond = get_loop_cond(loop_head, loop_exit);

  // If we didn't find a loop condition, don't change anything
  if(is_nil_expr(loop_cond))
  {
    std::cout << "**** WARNING: we couldn't find a loop condition for the"
              << " following loop, so we're not converting it." << std::endl
              << "Loop: ";
    loop.dump();
    return;
  }

  // Assume the loop condition before go into the loop
  assume_loop_cond_before_loop(loop_head, loop_cond);

  // Create the nondet assignments on the beginning of the loop
  make_nondet_assign(loop_head, loop);

  // Get original head again
  // Since we are using insert_swap to keep the targets, the
  // original loop head as shifted to after the assume cond
  while((++loop_head)->inductive_step_instruction)
    ;

  // Check if the loop exit needs to be updated
  // We must point to the assume that was inserted in the previous
  // transformation
  adjust_loop_head_and_exit(loop_head, loop_exit);
}

const expr2tc goto_k_inductiont::get_loop_cond(
  goto_programt::targett &loop_head,
  goto_programt::targett &loop_exit)
{
  // Let's not change the loop head
  goto_programt::targett tmp = loop_head;

  // Look for an loop condition
  while(!tmp->is_goto())
    ++tmp;

  // If we hit the loop's end and didn't find any loop condition
  // return a nil exprt
  if(tmp == loop_exit)
    return expr2tc();

  // We found the loop condition however it's inverted, so we
  // have to negate it before returning
  auto cond = tmp->guard;
  make_not(cond);
  return cond;
}

void goto_k_inductiont::make_nondet_assign(
  goto_programt::targett &loop_head,
  const loopst &loop)
{
  goto_programt dest;

  auto const &loop_vars = loop.get_loop_vars();

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
}

void goto_k_inductiont::assume_loop_cond_before_loop(
  goto_programt::targett &loop_head,
  expr2tc &loop_cond)
{
  if(is_true(loop_cond))
    return;

  goto_programt dest;
  assume_cond(loop_cond, dest);

  goto_function.body.insert_swap(loop_head, dest);
}

void goto_k_inductiont::assume_neg_loop_cond_after_loop(
  goto_programt::targett &loop_exit,
  expr2tc &loop_cond)
{
  goto_programt dest;
  expr2tc neg_loop_cond = loop_cond;
  make_not(neg_loop_cond);
  assume_cond(neg_loop_cond, dest);

  //  goto_programt::targett _loop_exit = loop_exit;
  //  ++_loop_exit;

  goto_function.body.insert_swap(loop_exit, dest);
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

// Duplicate the loop after loop_exit, but without the backward goto
void goto_k_inductiont::duplicate_loop_body(
  goto_programt::targett &loop_head,
  goto_programt::targett &loop_exit)
{
  goto_programt::targett _loop_exit = loop_exit;
  ++_loop_exit;

  // Iteration points will only be duplicated
  std::vector<goto_programt::targett> iteration_points;
  iteration_points.resize(2);

  if(_loop_exit != loop_head)
  {
    goto_programt::targett t_before = _loop_exit;
    t_before--;

    if(t_before->is_goto() && is_true(t_before->guard))
    {
      // no 'fall-out'
    }
    else
    {
      // guard against 'fall-out'
      goto_programt::targett t_goto = goto_function.body.insert(_loop_exit);

      t_goto->make_goto(_loop_exit);
      t_goto->location = _loop_exit->location;
      t_goto->function = _loop_exit->function;
      t_goto->guard = gen_true_expr();
    }
  }

  goto_programt::targett t_skip = goto_function.body.insert(_loop_exit);
  goto_programt::targett loop_iter = t_skip;

  t_skip->make_skip();
  t_skip->location = loop_head->location;
  t_skip->function = loop_head->function;

  // record the exit point of first iteration
  iteration_points[0] = loop_iter;

  // build a map for branch targets inside the loop
  std::map<goto_programt::targett, unsigned> target_map;

  {
    unsigned count = 0;
    for(goto_programt::targett t = loop_head; t != loop_exit; t++)
    {
      assert(t != goto_function.body.instructions.end());

      // Don't copy instructions inserted by the inductive-step
      // transformations
      if(t->inductive_step_instruction)
        continue;

      target_map[t] = count++;
    }
  }

  // we make k-1 copies, to be inserted before _loop_exit
  goto_programt copies;

  // make a copy
  std::vector<goto_programt::targett> target_vector;
  target_vector.reserve(target_map.size());

  for(goto_programt::targett t = loop_head; t != loop_exit; t++)
  {
    assert(t != goto_function.body.instructions.end());

    // Don't copy instructions inserted by the inductive-step
    // transformations
    if(t->inductive_step_instruction)
      continue;

    goto_programt::targett copied_t = copies.add_instruction();
    *copied_t = *t;
    target_vector.push_back(copied_t);
  }

  // record exit point of this copy
  iteration_points[1] = target_vector.back();

  // adjust the intra-loop branches
  for(unsigned i = 0; i < target_vector.size(); i++)
  {
    goto_programt::targett t = target_vector[i];

    for(auto &target : t->targets)
    {
      std::map<goto_programt::targett, unsigned>::const_iterator m_it =
        target_map.find(target);

      if(m_it != target_map.end()) // intra-loop?
      {
        assert(m_it->second < target_vector.size());
        target = target_vector[m_it->second];
      }
    }
  }

  // now insert copies before _loop_exit
  goto_function.body.insert_swap(loop_exit, copies);

  // remove skips
  remove_skip(goto_function.body);
}

// Convert assert into assumes on the original loop (don't touch the
// copy made on the last step)
void goto_k_inductiont::convert_assert_to_assume(
  goto_programt::targett &loop_head,
  goto_programt::targett &loop_exit)
{
  for(goto_programt::targett t = loop_head; t != loop_exit; t++)
    if(t->is_assert())
      t->type = ASSUME;
}

void goto_k_inductiont::assume_cond(const expr2tc &cond, goto_programt &dest)
{
  goto_programt tmp_e;
  goto_programt::targett e = tmp_e.add_instruction(ASSUME);
  e->inductive_step_instruction = true;
  e->guard = cond;
  dest.destructive_append(tmp_e);
}
