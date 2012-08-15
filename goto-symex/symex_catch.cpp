/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

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
    const irept::subt &exception_list=
      instruction.code.find("exception_list").get_sub();

    assert(exception_list.size()==instruction.targets.size());

    unsigned i=0;
    for(goto_programt::targetst::const_iterator
        it=instruction.targets.begin();
        it!=instruction.targets.end();
        it++, i++)
      frame.catch_map[exception_list[i].id()]=*it;
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
  const irept::subt &exceptions_thrown=
    instruction.code.find("exception_list").get_sub();

  // go through the call stack, beginning with the top
  for(goto_symex_statet::call_stackt::const_reverse_iterator
      s_it=cur_state->call_stack.rbegin();
      s_it!=cur_state->call_stack.rend();
      s_it++)
  {
    const goto_symex_statet::framet &frame=*s_it;

    if(frame.catch_map.empty()) continue;

    // throw without argument, we must rethrow last exception
    if(!exceptions_thrown.size())
    {
      if(last_throw != NULL && last_throw->code.find("exception_list").get_sub().size())
      {
        // get exception from last throw
        irept::subt::const_iterator e_it=last_throw->code.find("exception_list").get_sub().begin();

        // update current state exception list
        instruction.code.find("exception_list").get_sub().push_back((*e_it));
      }
      else
      {
        const std::string &msg="Trying to re-throw without last exception.";
        claim(false_exprt(), msg);
        return;
      }
    }

    for(irept::subt::const_iterator
        e_it=exceptions_thrown.begin();
        e_it!=exceptions_thrown.end();
        e_it++)
    {
      // Check if we can throw the exception
      if(has_throw_decl)
      {
        goto_symex_statet::framet::throw_list_sett::const_iterator
          s_it=frame.throw_list_set.find(e_it->id());

        if(s_it==frame.throw_list_set.end())
        {
          const std::string &msg="Trying to throw an exception of type " +
            e_it->id().as_string() + " but it's not allowed by declaration.";
          claim(false_exprt(), msg);
          return;
        }
      }

      // We can throw it, look on the map if we have a catch for it
      goto_symex_statet::framet::catch_mapt::const_iterator
        c_it=frame.catch_map.find(e_it->id());

      if(c_it!=frame.catch_map.end())
      {
        throw_target = (*c_it).second;
        has_throw_target=true;
        last_throw = &instruction; // save last throw
        return;
      }
      else // We don't have a catch for it
      {
        // Do we have an ellipsis?
        c_it=frame.catch_map.find("ellipsis");

        if(c_it!=frame.catch_map.end())
        {
          throw_target = (*c_it).second;
          has_throw_target=true;
          last_throw = &instruction; // save last throw
          return;
        }

        // An un-caught exception. Error
        const std::string &msg="Throwing an exception of type " +
          e_it->id().as_string() + " but there is not catch for it.";
        claim(false_exprt(), msg);
        return;
      }
    }
    last_throw = &instruction; // save last throw
  }
}

/*******************************************************************\

Function: goto_symext::symex_throw_decl

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_throw_decl()
{
  has_throw_decl = true;

  const goto_programt::instructiont &instruction= *cur_state->source.pc;

  // copy throw_list
  const irept::subt &throw_decl_list=
    instruction.code.find("throw_list").get_sub();

  goto_symex_statet::framet &frame=cur_state->call_stack.back();

  for(unsigned i=0; i<throw_decl_list.size(); ++i)
    frame.throw_list_set.insert(throw_decl_list[i].id());
}
