/*******************************************************************\

Module: SMT-LIB Frontend, parse tree

Author: CM Wintersteiger

\*******************************************************************/

#include "smt_parse_tree.h"

/*******************************************************************\

Function: smt_parse_treet::swap

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_parse_treet::swap(smt_parse_treet &smt_parse_tree)
{
  benchmarks.swap(smt_parse_tree.benchmarks);
  theories.swap(smt_parse_tree.theories);
  logics.swap(smt_parse_tree.logics);
}

/*******************************************************************\

Function: smt_parse_treet::clear

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_parse_treet::clear()
{
  benchmarks.clear();
  theories.clear();
  logics.clear();
}
