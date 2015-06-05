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

void goto_unwindt::create_function_loop(
  goto_programt::instructionst::iterator loop_head,
  goto_programt::instructionst::iterator loop_exit)
{
  goto_programt::instructionst::iterator it=loop_head;

  std::pair<goto_programt::instructiont, goto_programt>
    p(*loop_head, goto_programt());
  function_loops.insert(p);

  // We'll copy head and remove target number
  goto_programt::targett new_instruction=function_loops[p.first].add_instruction();
  *new_instruction=*it;

  // Remove head target number, there will be no backward loop
  // Duplicate code for the sake of OPTIMIZATION (no real gain on this, I guess)
  if(it == loop_head)
    new_instruction->target_number = unsigned(-1);

  // Next instruction
  ++it;

  // Copy the loop body
  while (it != loop_exit)
  {
    goto_programt::targett new_instruction=function_loops[p.first].add_instruction();
    *new_instruction=*it;
    ++it;
  }
}
