/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <stack>

#include "find_macros.h"

/*******************************************************************\

Function: find_macros

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void find_macros(
  const exprt &src,
  const namespacet &ns,
  find_macros_sett &dest)
{
  std::stack<const exprt *> stack;

  // use stack, these may be nested deeply  
  stack.push(&src);
  
  while(!stack.empty())
  {
    const exprt &e=*stack.top();
    stack.pop();
    
    if(e.id()=="symbol" ||
       e.id()=="next_symbol")
    {
      const irep_idt &identifier=e.identifier();
    
      const symbolt &symbol=ns.lookup(identifier);
      
      if(symbol.is_macro)
      {
        // inserted?
        if(dest.insert(identifier).second)
          stack.push(&symbol.value);
      }
    }
    else
    {
      forall_operands(it, e)
        stack.push(&(*it));
    }
  }
}
