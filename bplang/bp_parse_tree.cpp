/*******************************************************************\

Module: Boolean Program Parse Tree

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "bp_parse_tree.h"

/*******************************************************************\

Function: bp_parse_treet::swap

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bp_parse_treet::swap(bp_parse_treet &bp_parse_tree)
{
  bp_parse_tree.declarations.swap(declarations);
}

/*******************************************************************\

Function: bp_parse_treet::clear

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bp_parse_treet::clear()
{
  declarations.clear();
}
