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
  const goto_programt::instructiont &instruction=*cur_state->source.pc;

  if(instruction.targets.empty()) // The second catch, pop from the stack
  {
    // Copy the exception before pop
    goto_symex_statet::exceptiont exception=stack_catch.top();

    // Pop from the stack
    stack_catch.pop();

    // Increase the program counter
    cur_state->source.pc++;

    if(exception.has_throw_target)
    {
      // the next instruction is always a goto
      const goto_programt::instructiont &goto_instruction=*cur_state->source.pc;

      // Update target
      goto_instruction.targets.pop_back();
      goto_instruction.targets.push_back(exception.throw_target);

      exception.has_throw_target = false;
    }
  }
  else // The first catch, push it to the stack
  {
    goto_symex_statet::exceptiont exception;

    // copy targets
    const irept::subt &exception_list=
      instruction.code.find("exception_list").get_sub();

    assert(exception_list.size()==instruction.targets.size());

    // Fill the map with the catch type and the target
    unsigned i=0;
    for(goto_programt::targetst::const_iterator
        it=instruction.targets.begin();
        it!=instruction.targets.end();
        it++, i++)
    {
      exception.catch_map[exception_list[i].id()]=*it;
      exception.catch_order[exception_list[i].id()]=i;
    }

    // Stack it
    stack_catch.push(exception);

    // Increase the program counter
    cur_state->source.pc++;
  }
}

/*******************************************************************\

Function: goto_symext::symex_throw

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_symext::symex_throw()
{
  const goto_programt::instructiont &instruction= *cur_state->source.pc;

  // Get the list of exceptions thrown
  const irept::subt &exceptions_thrown=
    instruction.code.find("exception_list").get_sub();

  // Log
  std::cout << "*** Exception thrown of type "
    << exceptions_thrown.begin()->id().as_string()
    << " at file " << instruction.code.location().file()
    << " line " << instruction.code.location().line() << std::endl;

  // We check before iterate over the throw list to save time:
  // If there is no catch, we return an error
  if(!stack_catch.size())
  {
    if(!unexpected_handler())
      if(!terminate_handler())
      {
        // An un-caught exception. Error
        const std::string &msg="Throwing an exception of type " +
            exceptions_thrown.begin()->id().as_string() +
            " but there is not catch for it.";
        claim(false_exprt(), msg);
        return true;
      }
    return false;
  }

  // Get the list of catchs
  goto_symex_statet::exceptiont* except=&stack_catch.top();

  // Handle rethrows
  if(!handle_rethrow(exceptions_thrown, instruction))
    return true;

  // It'll be used for catch ordering when throwing
  // a derived object with multiple inheritance
  unsigned old_id_number=-1, new_id_number=0;

  codet catch_code("nil");
  for(irept::subt::const_iterator
      e_it=exceptions_thrown.begin();
      e_it!=exceptions_thrown.end();
      e_it++)
  {
    // Handle throw declarations
    switch(handle_throw_decl(except, e_it->id()))
    {
      case 0:
        return true;
        break;

      case 1:
        return false;
        break;

      case 2:
        break;

      default:
        assert(0);
        break;
    }

    // Search for a catch with a matching type
    goto_symex_statet::exceptiont::catch_mapt::const_iterator
      c_it=except->catch_map.find(e_it->id());

    // Do we have a catch for it?
    if(c_it!=except->catch_map.end() && !except->has_throw_target)
    {
      // We do!

      // Get current catch number and update if needed
      new_id_number = (*except->catch_order.find(e_it->id())).second;

      if(new_id_number < old_id_number)
      {
        update_throw_target(except,c_it->second,instruction.code);
        catch_code=c_it->second->code;
        catch_code.id(c_it->first);
      }

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
        c_it=except->catch_map.find("void_ptr");

        if(c_it!=except->catch_map.end() && !except->has_throw_target)
        {
          update_throw_target(except,c_it->second); // Make the jump to void*
          catch_code=c_it->second->code;
          catch_code.id(c_it->first);
        }
      }
      else
      {
        // Do we have an ellipsis?
        c_it=except->catch_map.find("ellipsis");

        if(c_it!=except->catch_map.end() && !except->has_throw_target)
        {
          update_throw_target(except,c_it->second);
          catch_code=c_it->second->code;
          catch_code.id(c_it->first);
        }
      }
    }
  }

  if(!except->has_throw_target)
  {
    // Call terminate handler before showing error message
    if(!terminate_handler())
    {
      // An un-caught exception. Error
      const std::string &msg="Throwing an exception of type " +
        exceptions_thrown.begin()->id().as_string() + " but there is not catch for it.";
      claim(false_exprt(), msg);
    }
  }
  else // save last throw for rethrow handling
  {
    last_throw = &instruction;

    // Log
    std::cout << "*** Caught by catch("
      << catch_code.id() << ") at file "
      << catch_code.location().file()
      << " line " << catch_code.location().line() << std::endl;
  }

  return true;
}

/*******************************************************************\

Function: goto_symext::terminate_handler

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_symext::terminate_handler()
{
  // We must look on the context if the user included exception lib
  const symbolt *tmp;
  bool is_included=ns.lookup("cpp::std::terminate()",tmp);

  // If it do, we must call the terminate function:
  // It'll call the current function handler
  if(!is_included) {
    codet terminate_function=to_code(tmp->value.op0());
    dereference(terminate_function,false);

    // We only call it if the user replaced the default one
    if(terminate_function.op1().identifier()=="cpp::std::default_terminate()")
      return false;

    // Call the function
    symex_function_call(to_code_function_call(terminate_function));
    return true;
  }

  // If it wasn't included, we do nothing. The error message will be
  // shown to the user as there is a throw without catch.
  return false;
}

/*******************************************************************\

Function: goto_symext::unexpected

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_symext::unexpected_handler()
{
  // We must look on the context if the user included exception lib
  const symbolt *tmp;
  bool is_included=ns.lookup("cpp::std::unexpected()",tmp);

  // If it do, we must call the unexpected function:
  // It'll call the current function handler
  if(!is_included) {
    codet unexpected_function=to_code(tmp->value.op0());
    dereference(unexpected_function,false);

    // We only call it if the user replaced the default one
    if(unexpected_function.op1().identifier()=="cpp::std::default_unexpected()")
      return false;

    // Call the function
    symex_function_call(to_code_function_call(unexpected_function));
    return true;
  }

  // If it wasn't included, we do nothing. The error message will be
  // shown to the user as there is a throw without catch.
  return false;
}

/*******************************************************************\

Function: goto_symext::update_throw_target

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::update_throw_target(goto_symex_statet::exceptiont* except,
    goto_programt::targett target, codet code)
{
  except->has_throw_target=true;
  except->throw_target=target;

  // We must update the value if it has operands
  if(code.operands().size())
    ns.lookup(target->code.op0().identifier()).value=code.op0();

  if(!options.get_bool_option("extended-try-analysis"))
  {
    statet::goto_state_listt &goto_state_list =
      cur_state->top().goto_state_map[target];

    goto_state_list.push_back(statet::goto_statet(*cur_state));

    statet::goto_statet &new_state = goto_state_list.back();
    cur_state->guard.make_false();
  }
}

/*******************************************************************\

Function: goto_symext::handle_throw_decl

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

int goto_symext::handle_throw_decl(goto_symex_statet::exceptiont* except,
  const irep_idt &id)
{
  // Check if we can throw the exception
  if(except->has_throw_decl)
  {
    goto_symex_statet::exceptiont::throw_list_sett::const_iterator
      s_it=except->throw_list_set.find(id);

    // Is it allowed?
    if(s_it==except->throw_list_set.end())
    {
      if(!unexpected_handler())
      {
        std::string msg=std::string("Trying to throw an exception ") +
            std::string("but it's not allowed by declaration.\n\n");
        msg += "  Exception type: " + id.as_string();
        msg += "\n  Allowed exceptions:";

        for(goto_symex_statet::exceptiont::throw_list_sett::iterator
            s_it1=except->throw_list_set.begin();
            s_it1!=except->throw_list_set.end();
            ++s_it1)
          msg+= "\n   - " + std::string((*s_it1).c_str());

        claim(false_exprt(), msg);
        return 0;
      }
      else
        return 1;
    }
  }
  return 2;
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
      cur_state->source.pc++;
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
  // Check if we have a previous try-block catch
  if(stack_catch.size())
  {
    const goto_programt::instructiont &instruction= *cur_state->source.pc;

    // Get throw_decl list
    const irept::subt &throw_decl_list=
      instruction.code.find("throw_list").get_sub();

    // Get to the correct try (always the last one)
    goto_symex_statet::exceptiont* except=&stack_catch.top();

    // Set the flag that this frame has throw list
    // This is important because we can have empty throw lists
    except->has_throw_decl=true;

    // Clear before insert new types
    except->throw_list_set.clear();

    // Copy throw list to the set
    for(unsigned i=0; i<throw_decl_list.size(); ++i)
      except->throw_list_set.insert(throw_decl_list[i].id());
  }
}
