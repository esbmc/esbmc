/*
 * loopst.cpp
 *
 *  Created on: Jun 30, 2015
 *      Author: mramalho
 */

#include "goto_loops.h"

#include <util/expr_util.h>

void goto_loopst::find_function_loops()
{
  std::map<unsigned int, goto_programt::instructionst::iterator> targets;

  for(goto_programt::instructionst::iterator
      it=goto_function.body.instructions.begin();
      it!=goto_function.body.instructions.end();
      it++)
  {
    // Record location number and targets
    if (it->is_target())
      targets[it->location_number] = it;

    // We found a loop, let's record its instructions
    if (it->is_backwards_goto())
    {
      assert(it->targets.size() == 1);
      create_function_loop(
        targets[(*it->targets.begin())->location_number], it);
    }
  }
}

void goto_loopst::create_function_loop(
  goto_programt::instructionst::iterator loop_head,
  goto_programt::instructionst::iterator loop_exit)
{
  goto_programt::instructionst::iterator it=loop_head;

  std::pair<goto_programt::targett, loopst>
    p(loop_head, loopst(goto_programt()));

  std::map<goto_programt::targett, loopst>::iterator it1 =
    function_loops.insert(p).first;

  // Set original iterators
  it1->second.set_original_loop_head(loop_head);
  it1->second.set_original_loop_exit(loop_exit);

  // Copy the loop body
  while (it != loop_exit)
  {
    goto_programt::targett new_instruction=
      it1->second.get_goto_program().add_instruction();

    // This should be done only when we're running k-induction
    // Maybe a flag on the class?
    if(it->is_assign())
    {
      const code_assign2t &assign = to_code_assign2t(it->code);
      add_loop_var(it1->second, migrate_expr_back(assign.target));
    }

    *new_instruction=*it;
    ++it;
  }

  // Finally, add the loop exit
  goto_programt::targett new_instruction=
    it1->second.get_goto_program().add_instruction();
  *new_instruction=*loop_exit;
}

void goto_loopst::output(std::ostream &out)
{
  for(function_loopst::iterator
      h_it=function_loops.begin();
      h_it!=function_loops.end();
      ++h_it)
  {
    unsigned n=h_it->first->location_number;

    out << n << " is head of { ";
    for(goto_programt::instructionst::iterator l_it=
        h_it->second.get_goto_program().instructions.begin();
        l_it!=h_it->second.get_goto_program().instructions.end(); ++l_it)
    {
      if(l_it!=h_it->second.get_goto_program().instructions.begin()) out << ", ";
        out << (*l_it).location_number;
    }
    out << " }\n";

    h_it->second.dump_loop_vars();
  }
}

void goto_loopst::add_loop_var(loopst &loop, const exprt& expr)
{
  if (expr.is_symbol() && expr.type().id() != "code")
  {
    if(check_var_name(expr))
      loop.add_var_to_loop(expr);
  }
  else
  {
    forall_operands(it, expr)
      add_loop_var(loop, *it);
  }
}
