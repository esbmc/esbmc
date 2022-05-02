/*******************************************************************\
 
Module: Convert goto functions to binary format and back (with irep
        hashing)
 
Author: CM Wintersteiger
 
Date: May 2007
 
\*******************************************************************/

#include <goto-programs/goto_function_serialization.h>
#include <goto-programs/goto_program_serialization.h>

void goto_function_serializationt::convert(
  const goto_functiont &function,
  std::ostream &out)
{
  if(function.body_available)
    gpconverter.convert(function.body, out);
}

void goto_function_serializationt::convert(std::istream &in, irept &funsymb)
{
  gpconverter.convert(in, funsymb);
  // don't forget to fix the functions type via the symbol table!
}
