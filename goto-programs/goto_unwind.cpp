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
  for(goto_programt::instructionst::const_iterator
      it=goto_function.body.instructions.begin();
      it!=goto_function.body.instructions.end();
      it++)
  {
    // We found a loop, let's record its instructions
    if (it->is_backwards_goto())
    {
      assert(it->targets.size() == 1);
      create_function_loop(
        (*it->targets.begin())->location_number, it->location_number);
    }
  }
}

void goto_unwindt::create_function_loop(
  unsigned int loop_head,
  unsigned int loop_exit)
{
  std::cout << "loop_head: " << loop_head << std::endl;
  std::cout << "loop_exit: " << loop_exit << std::endl;

  goto_programt::instructionst::iterator
    it=goto_function.body.instructions.begin();

  // Find loop head
  while (it->location_number != loop_head) it++;

  goto_programt loop_body;

  while (it->location_number != loop_exit)
  {
    goto_programt::targett new_instruction=loop_body.add_instruction();
    *new_instruction=*it;
    ++it;
  }
}
