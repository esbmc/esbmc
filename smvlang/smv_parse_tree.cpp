/*******************************************************************\

Module: SMV Parse Tree

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "smv_parse_tree.h"

/*******************************************************************\

Function: smv_parse_treet::swap

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smv_parse_treet::swap(smv_parse_treet &smv_parse_tree)
{
  smv_parse_tree.modules.swap(modules);
}

/*******************************************************************\

Function: smv_parse_treet::clear

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smv_parse_treet::clear()
{
  modules.clear();
}

/*******************************************************************\

Function: operator <<

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string to_string(smv_parse_treet::modulet::itemt::item_typet i)
{
  switch(i)
  {
  case smv_parse_treet::modulet::itemt::INVAR:    return "INVAR";
  case smv_parse_treet::modulet::itemt::TRANS:    return "TRANS";
  case smv_parse_treet::modulet::itemt::INIT:     return "INIT";
  case smv_parse_treet::modulet::itemt::SPEC:     return "SPEC";
  case smv_parse_treet::modulet::itemt::FAIRNESS: return "FAIRNESS";
  case smv_parse_treet::modulet::itemt::DEFINE:    return "DEFINE";  
  
  default:;
  }
  
  return "";
}
