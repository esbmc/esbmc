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
  const goto_programt::instructiont &instruction=*cur_state->source.pc;

  if(instruction.targets.empty()) // pop
  {
    if(cur_state->call_stack.empty())
      throw "catch-pop on empty call stack";

    if(cur_state->top().catch_map.empty())
      throw "catch-pop on function frame";

    // Copy the frame before pop
    goto_symex_statet::framet frame=cur_state->call_stack.back();

    // pop the stack frame
    cur_state->call_stack.pop_back();

    // Increase program counter
    cur_state->source.pc++;

    if(frame.has_throw_target)
    {
      // the next instruction is always a goto
      const goto_programt::instructiont &goto_instruction=*cur_state->source.pc;

      // Update target
      goto_instruction.targets.pop_back();
      goto_instruction.targets.push_back(frame.throw_target);

      frame.has_throw_target = false;
    }
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
      frame.catch_map[exception_list[i].id()]=*it;
      frame.catch_order[exception_list[i].id()]=i;
    }

    // Increase program counter
    cur_state->source.pc++;
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
  for(goto_symex_statet::call_stackt::reverse_iterator
      s_it=cur_state->call_stack.rbegin();
      s_it!=cur_state->call_stack.rend();
      s_it++)
  {
    goto_symex_statet::framet *frame=&(*s_it);

    if(frame->catch_map.empty()) continue;

    // Handle rethrows
    if(!handle_rethrow(exceptions_thrown, instruction))
      return;

    // It'll be used for catch ordering when throwing
    // a derived object with multiple inheritance
    unsigned old_id_number=-1, new_id_number=0;

    forall_names(e_it, throw_ref.exception_list)
    {
      // Handle throw declarations
      handle_throw_decl(frame, e_it->id());

      // We can throw! look on the map if we have a catch for the type thrown
      goto_symex_statet::framet::catch_mapt::const_iterator
        c_it=frame->catch_map.find(e_it->id());

      // Do we have a catch for it?
      if(c_it!=frame->catch_map.end() && !frame->has_throw_target)
      {
        // We do!

        // Get current catch number and update if needed
        new_id_number = (*frame->catch_order.find(e_it->id())).second;

        if(new_id_number < old_id_number)
          update_throw_target(frame, c_it);

        // Save old number id
        old_id_number = new_id_number;
      }
      else // We don't have a catch for it
      {
        // If it's a pointer, we must look for a catch(void*)
        if(e_it->id().as_string().find("_ptr") != std::string::npos)
        {
          // It's a pointer!

          // Do we have an void*?
          c_it=frame->catch_map.find("void_ptr");

          if(c_it!=frame->catch_map.end() && !frame->has_throw_target)
            update_throw_target(frame, c_it); // Make the jump to void*
        }
        else
        {
          // Do we have an ellipsis?
          c_it=frame->catch_map.find("ellipsis");

          if(c_it!=frame->catch_map.end() && !frame->has_throw_target)
            update_throw_target(frame, c_it);
        }
      }
    }

    if(!frame->has_throw_target)
    {
      // An un-caught exception. Error
      const std::string &msg="Throwing an exception of type " +
          e_it->id().as_string() + " but there is not catch for it.";
      claim(false_exprt(), msg);
    }

    // save last throw for rethrow handling
    last_throw = &instruction;
    return;
  }
}

/*******************************************************************\

Function: goto_symext::update_throw_target

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::update_throw_target(goto_symex_statet::framet* frame,
  goto_symex_statet::framet::catch_mapt::const_iterator c_it)
{
  frame->throw_target = (*c_it).second;
  frame->has_throw_target=true;
}

/*******************************************************************\

Function: goto_symext::handle_throw_decl

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::handle_throw_decl(goto_symex_statet::framet* frame,
  const irep_idt &id)
{
  // Check if we can throw the exception
  if(frame->has_throw_decl)
  {
    goto_symex_statet::framet::throw_list_sett::const_iterator
    s_it=frame->throw_list_set.find(id);

    if(s_it==frame->throw_list_set.end())
    {
      std::string msg=std::string("Trying to throw an exception ") +
          std::string("but it's not allowed by declaration.\n\n");
      msg += "  Exception type: " + id.as_string();
      msg += "\n  Allowed exceptions:";

      for(goto_symex_statet::framet::throw_list_sett::iterator
          s_it1=frame->throw_list_set.begin();
          s_it1!=frame->throw_list_set.end();
          ++s_it1)
        msg+= "\n   - " + std::string((*s_it1).c_str());

      claim(false_exprt(), msg);
      return;
    }
  }
}

/*******************************************************************\

Function: goto_symext::handle_rethrow

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_symext::handle_rethrow(irept::subt exceptions_thrown,
  const goto_programt::instructiont instruction)
{
  // throw without argument, we must rethrow last exception
  if(!exceptions_thrown.size())
  {
    if(last_throw != NULL && last_throw->code.find("exception_list").get_sub().size())
    {
      // get exception from last throw
      irept::subt::const_iterator e_it=last_throw->code.find("exception_list").get_sub().begin();

      // update current state exception list
      instruction.code.find("exception_list").get_sub().push_back((*e_it));

      return true;
    }
    else
    {
      const std::string &msg="Trying to re-throw without last exception.";
      claim(false_exprt(), msg);
      return false;
    }
  }
  return true;
}

/*******************************************************************\

Function: goto_symext::symex_throw_decl

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_throw_decl()
{
  const goto_programt::instructiont &instruction= *cur_state->source.pc;

  // Get throw list
  const irept::subt &throw_decl_list=
    instruction.code.find("throw_list").get_sub();

  // Get to the correct try (always the most external)
  goto_symex_statet::call_stackt::reverse_iterator
    s_it=cur_state->call_stack.rbegin();

  while(!(*s_it).catch_map.size()) ++s_it;

  // Set the flag that this frame has throw list
  // This is important because we can have empty throw lists
  (*s_it).has_throw_decl = true;

  // Clear before insert new types
  (*s_it).throw_list_set.clear();

  // Copy throw list to the set
  for(unsigned i=0; i<throw_decl_list.size(); ++i)
    (*s_it).throw_list_set.insert(throw_decl_list[i].id());
}
