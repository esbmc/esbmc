/*******************************************************************\

Module: Pointer Dereferencing

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "add_failed_symbols.h"

/*******************************************************************\

Function: add_failed_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void add_failed_symbol(symbolt &symbol, contextt &context)
{
  if(!symbol.lvalue) return;
  
  if(symbol.type.failed_symbol()!="")
    return;

  if(symbol.type.id()=="pointer")
  {
    symbolt new_symbol;
    new_symbol.lvalue=true;
    new_symbol.module=symbol.module;
    new_symbol.mode=symbol.mode;
    new_symbol.base_name=id2string(symbol.base_name)+"$object";
    new_symbol.name=id2string(symbol.name)+"$object";
    new_symbol.type=symbol.type.subtype();
    new_symbol.value.make_nil();
    
    symbol.type.failed_symbol(new_symbol.name);
    
    if(new_symbol.type.id()=="pointer")
      add_failed_symbol(new_symbol, context); // recursive call
        
    context.move(new_symbol);
  }
}

/*******************************************************************\

Function: add_failed_symbols

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void add_failed_symbols(contextt &context)
{
  Forall_symbols(it, context.symbols)
    add_failed_symbol(it->second, context);
}
