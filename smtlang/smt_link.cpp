/*******************************************************************\

Module: SMT-LIB Frontend, Linking

Author: CM Wintersteiger

\*******************************************************************/

#include "smt_link.h"
#include "smt_typecheck.h"

/*******************************************************************\

Function: smt_linkt::duplicate

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_linkt::duplicate(
  symbolt &in_context,
  symbolt &new_symbol)
{
  if(new_symbol.is_type!=in_context.is_type)
  {
    str << "class conflict on symbol `" << in_context.name
        << "'";
    throw 0;
  }

  if(new_symbol.is_type)
    duplicate_type(in_context, new_symbol);
  else
    duplicate_symbol(in_context, new_symbol);
}

/*******************************************************************\

Function: smt_linkt::duplicate_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_linkt::duplicate_type(
  symbolt &in_context,
  symbolt &new_symbol)
{
  if (in_context.is_extern && !new_symbol.is_extern)
  {
    in_context.type = new_symbol.type;
    in_context.is_extern = false;    
  }

  if(in_context.is_extern ^ new_symbol.is_extern) 
    return; // types will be different, but we don't care.
  
  if(in_context.type!=new_symbol.type)
  {
    str << "conflicting definitions for symbol `"
        << in_context.name << "'";
    throw 0;
  }
}

/*******************************************************************\

Function: smt_linkt::duplicate_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_linkt::duplicate_symbol(
  symbolt &in_context,
  symbolt &new_symbol)
{ 
  if (in_context.name==new_symbol.name &&
      in_context.name==smt_typecheckt::bsymn)
  {
    // merge benchmarks
    forall_operands(it, new_symbol.value) {
      in_context.value.copy_to_operands(*it);
    }
  }
  else
  {
    if (in_context.type!=new_symbol.type ||
        in_context.value!=new_symbol.value)
    {
      str << "conflicting definitions for symbol `"
          << in_context.name
          << "'";
      throw 0;
    }
  }  
}

/*******************************************************************\

Function: smt_linkt::link

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_linkt::typecheck()
{
  Forall_symbols(it, new_context.symbols)
    move(it->second);
}

/*******************************************************************\

Function: smt_linkt::move

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_linkt::move(symbolt &new_symbol)
{
  // try to add it
  symbolt *new_symbol_ptr;
  if(context.move(new_symbol, new_symbol_ptr))
    duplicate(*new_symbol_ptr, new_symbol);
}

/*******************************************************************\

Function: smt_link

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool smt_link(
  contextt &context,
  contextt &new_context,
  message_handlert &message_handler,
  const std::string &module)
{
  smt_linkt smt_link(context, new_context, module, message_handler);
  return smt_link.typecheck_main();
}
