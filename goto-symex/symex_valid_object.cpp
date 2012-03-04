/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <std_expr.h>

#include "goto_symex.h"
#include "dynamic_allocation.h"

static irep_idt get_object(const exprt &expr)
{
  if(expr.id()==exprt::symbol)
  {
    return to_symbol_expr(expr).get_identifier();
  }
  else if(expr.id()==exprt::member)
  {
    assert(expr.operands().size()==1);
    return get_object(expr.op0());
  }
  else if(expr.id()==exprt::index)
  {
    assert(expr.operands().size()==2);
    return get_object(expr.op0());
  }

  return "";
}

void goto_symext::replace_dynamic_allocation(exprt &expr)
{
  Forall_operands(it, expr)
    replace_dynamic_allocation(*it);

  if(expr.id()=="valid_object" || expr.id()=="deallocated_object")
  {
    assert(expr.operands().size()==1);
    assert(expr.op0().type().id()==typet::t_pointer);
    
    // check what we have
    if(expr.op0().id()==exprt::addrof ||
       expr.op0().id()=="implicit_address_of")
    {
      assert(expr.op0().operands().size()==1);
      exprt &object=expr.op0().op0();
      
      // see if that is a good one!
      const irep_idt identifier=get_object(object);
      
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
          expr.make_bool(is_valid_object(symbol));
          return; // done
        }
      }
    }

    // default behavior
    default_replace_dynamic_allocation(ns, expr);
  }
  else if(expr.id()=="dynamic_size")
  {
    // default behavior
    default_replace_dynamic_allocation(ns, expr);
  }
   else if(expr.id()=="invalid-pointer")
  {
    // default behavior
    default_replace_dynamic_allocation(ns, expr);
  }
  else if(expr.id()=="object_value")
  {
    assert(expr.operands().size()==1);
    expr.id(exprt::deref);
  }
}

bool goto_symext::is_valid_object(const symbolt &symbol)
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
