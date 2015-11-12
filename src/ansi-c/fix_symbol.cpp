/*******************************************************************\

Module: ANSI-C Linking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "fix_symbol.h"

/*******************************************************************\

Function: fix_symbolt::fix_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void fix_symbolt::fix_symbol(symbolt &symbol)
{
  type_mapt::const_iterator it=
    type_map.find(symbol.name);
  
  if(it!=type_map.end())
    symbol.name=it->second.id();
    
  replace(symbol.type);
  replace(symbol.value);
}

/*******************************************************************\

Function: fix_symbolt::fix_context

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void fix_symbolt::fix_context(contextt &context)
{
  for(type_mapt::const_iterator
      t_it=type_map.begin();
      t_it!=type_map.end();
      t_it++)
  {
    symbolst::iterator s_it=context.symbols.find(t_it->first);
    assert(s_it!=context.symbols.end());

    symbolt s=s_it->second;
    s.name=t_it->second.identifier();
    context.symbols.erase(s_it);
    context.move(s);
  }
}
