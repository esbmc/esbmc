/*
 * goto_unwind.cpp
 *
 *  Created on: Jun 3, 2015
 *      Author: mramalho
 */

#include <util/std_expr.h>

#include "goto_unwind.h"
#include "remove_skip.h"

void goto_unwind(
  contextt &context,
  goto_functionst& goto_functions,
  unsigned unwind,
  message_handlert& message_handler)
{
  Forall_goto_functions(it, goto_functions)
    if(it->second.body_available)
      goto_unwindt(
        context,
        it->first,
        goto_functions,
        it->second,
        unwind,
        message_handler);

  goto_functions.update();
}

void goto_unwindt::goto_unwind()
{
  // Full unwind the program
  for(function_loopst::reverse_iterator
    it = function_loops.rbegin();
    it != function_loops.rend();
    ++it)
  {
    unwind_program(goto_function.body, it);

    // remove skips
    remove_skip(goto_function.body);
  }
}

void goto_unwindt::unwind_program(
  goto_programt &goto_program,
  function_loopst::reverse_iterator loop)
{
  // Get loop exit goto number
  goto_programt::targett loop_exit = loop->get_original_loop_exit();

  // Increment pointer by 1, it will point to the first instruction
  // after the end of the loop
  loop_exit++;

  assert(unwind!=0);

  if(loop_exit!=goto_program.instructions.begin())
  {
    goto_programt::targett t_before=loop_exit;
    t_before--;

    if(t_before->is_goto() && is_true(t_before->guard))
    {
      // no 'fall-out'
    }
    else
    {
      // guard against 'fall-out'
      goto_programt::targett t_goto=goto_program.insert(loop_exit);

      t_goto->make_goto(loop_exit);
      t_goto->location=loop_exit->location;
      t_goto->function=loop_exit->function;
      t_goto->guard=true_expr;
    }
  }

  goto_programt::targett t_skip=goto_program.insert(loop_exit);
  goto_programt::targett loop_iter=t_skip;

  const goto_programt::targett loop_head = loop->get_original_loop_head();

  t_skip->make_skip();
  t_skip->location = loop_head->location;
  t_skip->function = loop_head->function;

  // build a map for branch targets inside the loop
  std::map<goto_programt::targett, unsigned> target_map;

  {
    unsigned count=0;
    for(goto_programt::targett t = loop->get_original_loop_head();
        t != loop_exit; t++, count++)
    {
      assert(t!=goto_program.instructions.end());
      target_map[t]=count;
    }
  }

  // re-direct any branches that go to loop_head to loop_iter

  for(goto_programt::targett t=loop_head;
      t!=loop_iter; t++)
  {
    assert(t!=goto_program.instructions.end());
    for(goto_programt::instructiont::targetst::iterator
        t_it=t->targets.begin();
        t_it!=t->targets.end();
        t_it++)
      if(*t_it==loop_head) *t_it=loop_iter;
  }

  // we make k-1 copies, to be inserted before loop_exit
  goto_programt copies;

  for(unsigned i=1; i<unwind; i++)
  {
    // make a copy
    std::vector<goto_programt::targett> target_vector;
    target_vector.reserve(target_map.size());

    for(goto_programt::targett t = loop->get_original_loop_head();
        t!=loop_exit; t++)
    {
      assert(t!=goto_program.instructions.end());
      goto_programt::targett copied_t=copies.add_instruction(*t);
      target_vector.push_back(copied_t);
    }

    // adjust the intra-loop branches

    for(unsigned i=0; i<target_vector.size(); i++)
    {
      goto_programt::targett t=target_vector[i];

      for(goto_programt::instructiont::targetst::iterator
          t_it=t->targets.begin();
          t_it!=t->targets.end();
          t_it++)
      {
        std::map<goto_programt::targett, unsigned>::const_iterator
          m_it=target_map.find(*t_it);
        if(m_it!=target_map.end()) // intra-loop?
        {
          assert(m_it->second<target_vector.size());
          *t_it=target_vector[m_it->second];
        }
      }
    }
  }

  assert(copies.instructions.size()==(unwind-1)*target_map.size());

  // now insert copies before loop_exit
  goto_program.destructive_insert(loop_exit, copies);
}
