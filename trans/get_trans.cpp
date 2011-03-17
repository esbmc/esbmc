/*******************************************************************\

Module: Get Transition System out of Context

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "get_trans.h"

/*******************************************************************\

Function: get_trans

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

const transt &get_trans(
  const namespacet &ns,
  const irep_idt &module)
{
  const symbolt &symbol=ns.lookup(module);

  if(symbol.value.id()!="trans")
    throw "module `"+id2string(module)+"' is not a transition system";

  return to_trans(symbol.value);
}
