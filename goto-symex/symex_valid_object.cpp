/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <std_expr.h>

#include "goto_symex.h"
#include "dynamic_allocation.h"

static irep_idt get_object(const expr2tc &expr)
{
  if (is_symbol2t(expr))
  {
    return to_symbol2t(expr).name;
  }
  else if (is_member2t(expr))
  {
    return get_object(to_member2t(expr).source_value);
  }
  else if (is_index2t(expr))
  {
    return get_object(to_index2t(expr).source_value);
  }

  return "";
}

void goto_symext::replace_dynamic_allocation(expr2tc &expr)
{

  std::vector<expr2tc *> operands;
  expr.get()->list_operands(operands);
  for (std::vector<expr2tc *>::const_iterator it = operands.begin();
       it != operands.end(); it++)
    replace_dynamic_allocation(**it);

  if (is_valid_object2t(expr) || is_deallocated_obj2t(expr))
  {
    expr2tc &obj_ref = (is_valid_object2t(expr))
                        ? to_valid_object2t(expr).value
                        : to_deallocated_obj2t(expr).value;

    // check what we have
    if (is_address_of2t(obj_ref))
    {
      expr2tc &obj_operand = to_address_of2t(obj_ref).ptr_obj;
      
      // see if that is a good one!
      const irep_idt identifier = get_object(obj_operand);
      
      if(identifier!="")
      {        
        const irep_idt l0_identifier=
          cur_state->get_original_name(identifier);

        const symbolt &symbol=ns.lookup(l0_identifier);
        
        // dynamic?
        if(symbol.type.dynamic())
        {
          // TODO
        }
        else
        {
          expr = expr2tc(new constant_bool2t(is_valid_object(symbol)));
          return; // done
        }
      }
    }

    // default behavior
    default_replace_dynamic_allocation(expr);
  }
  else if (is_dynamic_size2t(expr))
  {
    // default behavior
    default_replace_dynamic_allocation(expr);
  }
  else if (is_invalid_pointer2t(expr))
  {
    // default behavior
    default_replace_dynamic_allocation(expr);
  }
}

bool
goto_symext::is_valid_object(const symbolt &symbol)
{
  if(symbol.static_lifetime) return true; // global
  
  // dynamic?
  if(symbol.type.dynamic())
    return false;

  // current location?
  if(cur_state->source.is_set &&
     cur_state->source.pc->local_variables.find(symbol.name)!=
     cur_state->source.pc->local_variables.end())
    return true;

  // search call stack frames
  for(goto_symext::statet::call_stackt::const_iterator
      it=cur_state->call_stack.begin();
      it!=cur_state->call_stack.end();
      it++)
    if(it->calling_location.is_set &&
       it->calling_location.pc->local_variables.find(symbol.name)!=
       it->calling_location.pc->local_variables.end())
      return true;

  return false;
}
