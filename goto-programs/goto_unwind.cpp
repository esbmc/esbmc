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
  unsigned unwind,
  const namespacet& ns,
  message_handlert& message_handler)
{
  Forall_goto_functions(it, goto_functions)
    if(it->second.body_available)
      goto_unwindt(it->second, unwind, ns, message_handler);
}

void goto_unwindt::goto_unwind()
{
  handle_nested_loops();
}

void goto_unwindt::handle_nested_loops()
{
  for(function_loopst::iterator
    it = function_loops.begin();
    it != function_loops.end();
    ++it)
  {
    // Possible superset that we're looking in
    handle_nested_loops_rec(it, false);
  }

  // Clean up
  for(function_loopst::iterator
      it = function_loops.begin();
      it != function_loops.end();
      ++it)
    {
      if(it->second.empty())
        function_loops.erase(it);
    }
}

void goto_unwindt::handle_nested_loops_rec(
  function_loopst::iterator superset,
  bool rec)
{
  // If the set is empty, then we already expanded it
  if(superset->second.empty())
    return;

  goto_programt::instructionst::iterator loop_head =
    superset->second.instructions.begin();

  goto_programt::instructionst::iterator loop_exit =
    superset->second.instructions.end();

  // Alright, can we get to the end of it, without finding another loop?
  while(++loop_head != loop_exit)
  {
    function_loopst::iterator found =
      function_loops.find(loop_head->location_number);

    if(found != function_loops.end())
    {
      // Nested loops, handle them before finish the current one
      handle_nested_loops_rec(found, true);

      // Insert copies before the current upper loop
      if(!tmp_goto_program.empty())
      {
        // Insert GOTO instructions
        superset->second.destructive_insert(loop_head, tmp_goto_program);

        // Get the number of the nested loop exit
        unsigned nested_loop_exit =
          (--found->second.instructions.end())->location_number;

        // Cleanup set
        found->second.clear();

        // Finally, remove the last loop and one extra instruction
        // which is the backward goto
        while(loop_head->location_number <= (nested_loop_exit+1))
          loop_head = superset->second.instructions.erase(loop_head);
      }
    }
  }

  // Only create copies of nested loops
  if(!rec)
    return;

  goto_programt copies;

  // Create k copies of the loop
  for(unsigned i=0; i < unwind; ++i)
  {
    for(goto_programt::instructionst::const_iterator
        l_it=superset->second.instructions.begin();
        l_it!=superset->second.instructions.end();
        ++l_it)
    {
      goto_programt::targett copied_t=copies.add_instruction();
      *copied_t=*l_it;
    }
  }

  // Save copy to be added to upper loop
  tmp_goto_program.destructive_append(copies);
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

  std::pair<unsigned, goto_programt>
    p(loop_head->location_number, goto_programt());
  function_loops.insert(p);

  // We'll copy head and remove target number
  goto_programt::targett new_instruction=
    function_loops[p.first].add_instruction();
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
    goto_programt::targett new_instruction=
      function_loops[p.first].add_instruction();
    *new_instruction=*it;
    ++it;
  }
}

void goto_unwindt::output(std::ostream &out)
{
  for(function_loopst::const_iterator h_it=function_loops.begin();
      h_it!=function_loops.end(); ++h_it)
  {
    unsigned n=h_it->first;

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
