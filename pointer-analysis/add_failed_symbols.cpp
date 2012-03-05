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

void add_failed_symbol(symbolt &symbol, contextt &context, namespacet &ns)
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

    // Rename symbol type; in the new type renaming model, contents of pointer
    // types aren't renamed, so it has to happen when they're "dereference"
    // like here.
    const typet derefedtype = ns.follow(new_symbol.type);
    new_symbol.type = derefedtype;
    
    symbol.type.failed_symbol(new_symbol.name);
    
    if(new_symbol.type.id()=="pointer")
      add_failed_symbol(new_symbol, context, ns); // recursive call
        
    context.move(new_symbol);
  }
}

/*******************************************************************\

Function: add_failed_symbols

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void add_failed_symbols(contextt &context, namespacet &ns)
{
  Forall_symbols(it, context.symbols)
    add_failed_symbol(it->second, context, ns);
}
