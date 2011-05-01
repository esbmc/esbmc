/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <std_expr.h>

#include "goto_symex.h"

/*******************************************************************\

Function: get_object

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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

/*******************************************************************\

Function: goto_symext::do_valid_object

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::replace_dynamic_allocation(
  const statet &state,
  exprt &expr)
{
  Forall_operands(it, expr)
    replace_dynamic_allocation(state, *it);

  if(expr.id()=="valid_object" || expr.id()=="deallocated_object")
  {
	//std::cout << "expr.pretty(): " << expr.pretty() << std::endl;
    assert(expr.operands().size()==1);
    assert(expr.op0().type().id()=="pointer");
    
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
          state.get_original_name(identifier);

        const symbolt &symbol=ns.lookup(l0_identifier);
        
        // dynamic?
        if(symbol.type.get_bool("#dynamic"))
        {
          // TODO
        }
        else
        {
          expr.make_bool(is_valid_object(state, symbol));
          return; // done
        }
      }
    }

    // default behavior
    basic_symext::replace_dynamic_allocation(state, expr);
  }
  else if(expr.id()=="dynamic_size")
  {
    // default behavior
    basic_symext::replace_dynamic_allocation(state, expr);
  }
  else if(expr.id()=="object_value")
  {
    assert(expr.operands().size()==1);
    expr.id(exprt::deref);
  }
}

/*******************************************************************\

Function: goto_symext::is_valid_object

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_symext::is_valid_object(
  const statet &state,
  const symbolt &symbol)
{
  if(symbol.static_lifetime) return true; // global
  
  // dynamic?
  if(symbol.type.get_bool("#dynamic"))
    return false;

  // current location?
  if(state.source.is_set &&
     state.source.pc->local_variables.find(symbol.name)!=
     state.source.pc->local_variables.end())
    return true;

  // search call stack frames
  for(goto_symext::statet::call_stackt::const_iterator
      it=state.call_stack.begin();
      it!=state.call_stack.end();
      it++)
    if(it->calling_location.is_set &&
       it->calling_location.pc->local_variables.find(symbol.name)!=
       it->calling_location.pc->local_variables.end())
      return true;

  return false;
}
