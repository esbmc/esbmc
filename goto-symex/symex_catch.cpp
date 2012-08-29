/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>
#include "goto_symex.h"

#include <simplify_expr.h>

#include "goto_symex.h"

/*******************************************************************\

Function: goto_symext::symex_catch

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_catch()
{
  // there are two variants: 'push' and 'pop'
  const goto_programt::instructiont &instruction= *cur_state->source.pc;

  if(instruction.targets.empty()) // pop
  {
    if(cur_state->call_stack.empty())
      throw "catch-pop on empty call stack";

    if(cur_state->top().catch_map.empty())
      throw "catch-pop on function frame";

    has_catch=true;

    // pop the stack frame
    cur_state->call_stack.pop_back();
  }
  else // push
  {
    cur_state->call_stack.push_back(goto_symex_statet::framet(cur_state->source.thread_nr));
    goto_symex_statet::framet &frame=cur_state->call_stack.back();

    // copy targets
    const code_cpp_catch2t &catch_ref = to_code_cpp_catch2t(instruction.code);

    assert(catch_ref.exception_list.size()==instruction.targets.size());

    unsigned i=0;
    for(goto_programt::targetst::const_iterator
        it=instruction.targets.begin();
        it!=instruction.targets.end();
        it++, i++)
    {
      const irep_idt &entry = catch_ref.exception_list[i];
      frame.catch_map[entry]=*it;
      //std::cout << "exception_list[i].id(): " << exception_list[i].id() << std::endl;
      //std::cout << "(*it)->code: " << (*it)->code << std::endl;
    }
  }
}

/*******************************************************************\

Function: goto_symext::symex_throw

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_throw()
{
  const goto_programt::instructiont &instruction= *cur_state->source.pc;

  // get the list of exceptions thrown
  const code_cpp_throw2t &throw_ref = to_code_cpp_throw2t(instruction.code);

  // go through the call stack, beginning with the top

  for(goto_symex_statet::call_stackt::const_reverse_iterator
      s_it=cur_state->call_stack.rbegin();
      s_it!=cur_state->call_stack.rend();
      s_it++)
  {
    const goto_symex_statet::framet &frame=*s_it;

    if(frame.catch_map.empty()) continue;

    if(!throw_ref.exception_list.size())
    // throw without argument, we must rethrow last exception
    {
      if(last_throw != NULL &&
         to_code_cpp_throw2t(last_throw->code).exception_list.size())
      {
        // get exception from last throw
        const code_cpp_throw2t &last_throw_ref =
          to_code_cpp_throw2t(last_throw->code);

        // update current state exception list
        code_cpp_throw2t &mutable_throw =
          const_cast<code_cpp_throw2t &>(throw_ref);
        mutable_throw.exception_list.push_back(
            *last_throw_ref.exception_list.begin());
      }
      else
      {
        const std::string &msg="Trying to re-throw without last exception";
        claim(false_expr, msg);
        return;
      }
    }

    forall_names(e_it, throw_ref.exception_list)
    {
      goto_symex_statet::framet::catch_mapt::const_iterator
      c_it=frame.catch_map.find(*e_it);

      if(c_it!=frame.catch_map.end())
      {
    	  throw_target = (*c_it).second;
    	  has_throw_target=true;
#if 0
        goto_programt::const_targett goto_target =
        		(*c_it).second;

        goto_programt::const_targett new_state_pc, state_pc;

        new_state_pc = goto_target; // goto target instruction
        state_pc = cur_state->source.pc;
        state_pc++; // next instruction

        cur_state->source.pc = state_pc;

        new_state_pc->guard.make_false();

        // put into state-queue
        statet::goto_state_listt &goto_state_list =
          cur_state->top().goto_state_map[new_state_pc];

        goto_state_list.push_back(statet::goto_statet(*cur_state));
        statet::goto_statet &new_state = goto_state_list.back();

        cur_state->guard.make_true();
        has_throw_target = true;
#endif
        last_throw = const_cast<goto_programt::instructiont *>
                               (&instruction); // save last throw
        return;
      }
      else
      {
        // Do we have an ellipsis?
        c_it=frame.catch_map.find("ellipsis");

        if(c_it!=frame.catch_map.end())
        {
          throw_target = (*c_it).second;
          has_throw_target=true;
          last_throw = const_cast<goto_programt::instructiont *>
                                 (&instruction); // save last throw
          return;
        }

        // An un-caught exception. Behaves like assume(0);
        cur_state->guard.add(false_expr);
        expr2tc tmp = cur_state->guard.as_expr();
        target->assumption(cur_state->guard.as_expr(), tmp, cur_state->source);
      }
    }
    last_throw = const_cast<goto_programt::instructiont *>
                           (&instruction); // save last throw
  }
}
