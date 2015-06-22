/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "ansi_c_parse_tree.h"

/*******************************************************************\

Function: ansi_c_parse_treet::swap

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ansi_c_parse_treet::swap(ansi_c_parse_treet &ansi_c_parse_tree)
{
  ansi_c_parse_tree.declarations.swap(declarations);
}

/*******************************************************************\

Function: ansi_c_parse_treet::clear

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ansi_c_parse_treet::clear()
{
  declarations.clear();
}

