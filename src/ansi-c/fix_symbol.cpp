/*******************************************************************\

Module: ANSI-C Linking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/fix_symbol.h>

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
    symbolt* symb = context.find_symbol(t_it->first);
    assert(symb != nullptr);

    symbolt s = *symb;
    s.name = t_it->second.identifier();
    context.erase_symbol(t_it->first);
    context.move(s);
  }
}
