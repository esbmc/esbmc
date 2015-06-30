/*
 * loopst.cpp
 *
 *  Created on: Jun 30, 2015
 *      Author: mramalho
 */

#include "goto_loops.h"

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

  std::pair<goto_programt::targett, goto_programt>
    p(loop_head, goto_programt());
  function_loops.insert(p);

  // Copy the loop body
  while (it != loop_exit)
  {
    goto_programt::targett new_instruction=
      function_loops[p.first].add_instruction();
    *new_instruction=*it;
    ++it;
  }
}

void goto_loopst::output(std::ostream &out)
{
  for(function_loopst::const_iterator
      h_it=function_loops.begin();
      h_it!=function_loops.end();
      ++h_it)
  {
    unsigned n=h_it->first->location_number;

    out << n << " is head of { ";
    for(goto_programt::instructionst::const_iterator l_it=
        h_it->second.instructions.begin();
        l_it!=h_it->second.instructions.end(); ++l_it)
    {
      if(l_it!=h_it->second.instructions.begin()) out << ", ";
      out << (*l_it).location_number;
    }
    out << " }\n";
  }
}
