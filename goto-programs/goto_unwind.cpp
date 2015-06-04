/*
 * goto_unwind.cpp
 *
 *  Created on: Jun 3, 2015
 *      Author: mramalho
 */

#include <util/std_expr.h>

#include "goto_unwind.h"

void goto_unwind(
  goto_functionst& goto_functions,
  const namespacet& ns,
  message_handlert& message_handler)
{
  Forall_goto_functions(it, goto_functions)
    if(it->second.body_available)
      goto_unwindt(it->second, ns, message_handler);
}

void goto_unwindt::goto_unwind_rec()
{
}

void goto_unwindt::find_function_loops()
{
  std::map<unsigned int, goto_programt::instructionst::iterator> targets;

  for(goto_programt::instructionst::iterator
      it=goto_function.body.instructions.begin();
      it!=goto_function.body.instructions.end();
      it++)
  {
    // Record location number and targets
    if (it->is_goto())
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

void goto_unwindt::create_function_loop(
  goto_programt::instructionst::iterator loop_head,
  goto_programt::instructionst::iterator loop_exit)
{
  goto_programt::instructionst::iterator it=loop_head;

  // Copy the loop body
  goto_programt loop_body;
  while (it != loop_exit)
  {
    goto_programt::targett new_instruction=loop_body.add_instruction();
    *new_instruction=*it;
    new_instruction->target_number = unsigned(-1);
    new_instruction->targets.clear();
    ++it;
  }

  // Update labels
  loop_body.update();

  std::pair<goto_programt::instructiont, goto_programt> p(*loop_head, goto_programt());
  function_loops.insert(p);

  // Save on the map
  function_loops[*loop_head].copy_from(loop_body);
}
