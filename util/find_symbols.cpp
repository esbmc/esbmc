/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "find_symbols.h"

/*******************************************************************\

Function: find_symbols

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void find_symbols(
  const exprt &src,
  find_symbols_sett &dest)
{
  find_symbols(src, dest, true, true);
}

/*******************************************************************\

Function: find_symbols

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void find_symbols(
  const exprt &src,
  find_symbols_sett &dest,
  bool current,
  bool next)
{
  if((src.id()=="symbol" && current) ||
     (src.id()=="next_symbol" && next))
    dest.insert(src.get("identifier"));
  else
  {
    forall_operands(it, src)
      find_symbols(*it, dest, current, next);
  }
}

/*******************************************************************\

Function: has_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool has_symbol(
  const exprt &src,
  const find_symbols_sett &symbols,
  bool current,
  bool next)
{
  if((src.id()=="symbol" && current) ||
     (src.id()=="next_symbol" && next))
    return symbols.count(src.get("identifier"))!=0;
  else
  {
    forall_operands(it, src)
      if(has_symbol(*it, symbols, current, next))
        return true;
  }
  
  return false;
}

/*******************************************************************\

Function: has_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool has_symbol(
  const exprt &src,
  const find_symbols_sett &symbols)
{
  return has_symbol(src, symbols, true, true);
}

/*******************************************************************\

Function: find_symbols

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void find_symbols(
  const exprt &src,
  std::set<exprt> &dest)
{
  if(src.id()=="symbol" || src.id()=="next_symbol")
    dest.insert(src);
  else
  {
    forall_operands(it, src)
      find_symbols(*it, dest);
  }
}
