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

  if(instruction.targets.empty()) // The second catch, pop from the stack
  {
    // Copy the exception before pop
    goto_symex_statet::exceptiont exception=stack_catch.top();

    // Pop from the stack
    stack_catch.pop();

    // Increase the program counter
    cur_state->source.pc++;
  }
  else // The first catch, push it to the stack
  {
    goto_symex_statet::exceptiont exception;

    // copy targets
    const code_cpp_catch2t &catch_ref = to_code_cpp_catch2t(instruction.code);

    assert(catch_ref.exception_list.size()==instruction.targets.size());

    // Fill the map with the catch type and the target
    unsigned i=0;
    for(goto_programt::targetst::const_iterator
        it=instruction.targets.begin();
        it!=instruction.targets.end();
        it++, i++)
    {
      exception.catch_map[catch_ref.exception_list[i]]=*it;
      exception.catch_order[catch_ref.exception_list[i]]=i;
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
  irep_idt catch_name = "missing";
  const goto_programt::const_targett *catch_insn = NULL;
  const goto_programt::instructiont &instruction= *cur_state->source.pc;

  // get the list of exceptions thrown
  const code_cpp_throw2t &throw_ref = to_code_cpp_throw2t(instruction.code);
  const std::vector<irep_idt> exceptions_thrown = throw_ref.exception_list;

  // Handle rethrows
  if(handle_rethrow(throw_ref.operand, instruction))
    return true;

  // Save the throw
  last_throw = const_cast<goto_programt::instructiont*>(&instruction);

  // Log
  std::cout << "*** Exception thrown of type "
    << exceptions_thrown.begin()->as_string()
    << " at file " << instruction.location.file()
    << " line " << instruction.location.line() << std::endl;

  // We check before iterate over the throw list to save time:
  // If there is no catch, we return an error
  if(!stack_catch.size())
  {
    if(!unexpected_handler())
    {
      if(!terminate_handler())
      {
        // An un-caught exception. Error
        const std::string &msg="Throwing an exception of type " +
            exceptions_thrown.begin()->as_string() +
            " but there is not catch for it.";
        claim(gen_false_expr(), msg);
        return true;
      }
    }
    return false;
  }

  // Get the list of catchs
  goto_symex_statet::exceptiont* except=&stack_catch.top();

  // It'll be used for catch ordering when throwing
  // a derived object with multiple inheritance
  unsigned old_id_number=-1, new_id_number=0;

  forall_names(e_it, throw_ref.exception_list)
  {
    // Handle throw declarations
    switch(handle_throw_decl(except, *e_it))
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
      c_it=except->catch_map.find(*e_it);

    // Do we have a catch for it?
    if(c_it!=except->catch_map.end())
    {
      // We do!

      // Get current catch number and update if needed
      new_id_number = (*except->catch_order.find(*e_it)).second;

      if(new_id_number < old_id_number)
      {
        update_throw_target(except,c_it->second,instruction.code);
        catch_insn = &c_it->second;
        catch_name = c_it->first;
      }

      // Save old number id
      old_id_number = new_id_number;
    }
    else // We don't have a catch for it
    {
      // If it's a pointer, we must look for a catch(void*)
      if(e_it->as_string().find("_ptr") != std::string::npos)
      {
        // It's a pointer!

        // Do we have an void*?
        c_it=except->catch_map.find("void_ptr");

        if(c_it!=except->catch_map.end())
        {
          // Make the jump to void*
          update_throw_target(except, c_it->second,instruction.code);
          catch_insn = &c_it->second;
          catch_name = c_it->first;
        }
      }
      else
      {
        // Do we have an ellipsis?
        c_it=except->catch_map.find("ellipsis");

        if(c_it!=except->catch_map.end())
        {
          update_throw_target(except,c_it->second,instruction.code);
          catch_insn = &c_it->second;
          catch_name = c_it->first;
        }
      }
    }
  }

  if (catch_insn == NULL) {
    // No catch for type, void, or ellipsis
    // Call terminate handler before showing error message
    if(!terminate_handler())
    {
      // An un-caught exception. Error
      const std::string &msg="Throwing an exception of type " +
        exceptions_thrown.begin()->as_string() + " but there is not catch for it.";
      claim(gen_false_expr(), msg);
      // Ensure no further execution along this path.
      cur_state->guard.make_false();
    }

    return false;
  }


  // Log
  std::cout << "*** Caught by catch("
    << catch_name << ") at file "
    << (*catch_insn)->location.file()
    << " line " << (*catch_insn)->location.line() << std::endl;

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

    // We only call it if the user replaced the default one
    if(terminate_function.op1().identifier()=="cpp::std::default_terminate()")
      return false;

    // Call the function
    expr2tc da_funk;
    migrate_expr(terminate_function, da_funk);
    symex_function_call(da_funk);
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
  // Look if we already on the unexpected flow
  // If true, we shouldn't call the unexpected handler again
  if(inside_unexpected)
    return false;

  // We must look on the context if the user included exception lib
  const symbolt *tmp;
  bool is_included=ns.lookup("cpp::std::unexpected()",tmp);

  // If it do, we must call the unexpected function:
  // It'll call the current function handler
  if(!is_included) {
    expr2tc the_call;
    code_function_callt unexpected_function;
    unexpected_function.function()=symbol_expr(*tmp);
    migrate_expr(unexpected_function, the_call);

    // We only call it if the user replaced the default one
    if (to_symbol2t(to_code_function_call2t(the_call).function).thename ==
        "cpp::std::default_unexpected()")
      return false;

    // Indicate there we're inside the unexpected flow
    inside_unexpected=true;

    // Call the function
    symex_function_call(the_call);
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

void goto_symext::update_throw_target(goto_symex_statet::exceptiont* except
                                      __attribute__((unused)),
                                      goto_programt::const_targett target,
                                      const expr2tc &code)
{

  // Something is going to catch, therefore we need something to be caught.
  // Assign that something to a variable and make records so that it's merged
  // into the right place in the future.
  assert(!is_nil_expr(code));
  code_cpp_throw2tc throw_insn(code);

  // Generate a name to assign this to.
  symbol2tc thrown_obj(throw_insn->operand->type,
                       irep_idt("symex_throw::thrown_obj"));
  expr2tc operand(throw_insn->operand);
  guardt g;
  symex_assign_symbol(thrown_obj, operand, g);

  // Now record that value for future reference.
  cur_state->rename(thrown_obj);

  // Target is, as far as I can tell, always a declaration of the variable
  // that the thrown obj ends up in, and is followed by a (blank) assignment
  // to it. So point at the next insn.
  assert(is_code_decl2t(target->code));
  target++;
  assert(is_code_assign2t(target->code));

  // Signal assignment code to fetch the thrown object and rewrite the
  // assignment, assigning the thrown obj to the local variable.
  thrown_obj_map[target] = thrown_obj;

  if(!options.get_bool_option("extended-try-analysis"))
  {
    // Search backwards through stack frames, looking for the frame that
    // contains the function containing the target instruction.
    goto_symex_statet::call_stackt::reverse_iterator i;
    for (i = cur_state->call_stack.rbegin();
         i != cur_state->call_stack.rend(); i++) {
      if (i->function_identifier == target->function) {
        statet::goto_state_listt &goto_state_list = i->goto_state_map[target];

        goto_state_list.push_back(statet::goto_statet(*cur_state));
        cur_state->guard.make_false();
        break;
      }
    }

    assert(i != cur_state->call_stack.rend() && "Target instruction in throw "
           "handler not in any function frame on the stack");
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

        claim(gen_false_expr(), msg);
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

bool goto_symext::handle_rethrow(const expr2tc &operand,
  const goto_programt::instructiont &instruction)
{
  // throw without argument, we must rethrow last exception
  if(is_nil_expr(operand))
  {
    if(last_throw != NULL && to_code_cpp_throw2t(last_throw->code).exception_list.size())
    {
      // get exception from last throw
      std::vector<irep_idt>::const_iterator e_it =
        to_code_cpp_throw2t(last_throw->code).exception_list.begin();

      // update current state exception list
      goto_programt::instructiont &mutable_ref =
        const_cast<goto_programt::instructiont &>(instruction);
      to_code_cpp_throw2t(mutable_ref.code).exception_list.push_back((*e_it));

      return true;
    }
    else
    {
      const std::string &msg="Trying to re-throw without last exception.";
      claim(gen_false_expr(), msg);
      return true;
    }
  }
  return false;
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

    // Get throw list
    const std::vector<irep_idt> &throw_decl_list =
      to_code_cpp_throw_decl2t(instruction.code).exception_list;

    // Get to the correct try (always the last one)
    goto_symex_statet::exceptiont* except=&stack_catch.top();

    // Set the flag that this frame has throw list
    // This is important because we can have empty throw lists
    except->has_throw_decl=true;

    // Clear before insert new types
    except->throw_list_set.clear();

    // Copy throw list to the set
    for(unsigned i=0; i<throw_decl_list.size(); ++i)
      except->throw_list_set.insert(throw_decl_list[i]);
  }
}
