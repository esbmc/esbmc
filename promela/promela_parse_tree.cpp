/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "promela_parse_tree.h"

/*******************************************************************\

Function: promela_parse_treet::swap

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void promela_parse_treet::swap(promela_parse_treet &promela_parse_tree)
{
  promela_parse_tree.declarations.swap(declarations);
}

/*******************************************************************\

Function: promela_parse_treet::clear

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void promela_parse_treet::clear()
{
  declarations.clear();
}

