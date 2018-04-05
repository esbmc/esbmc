/*******************************************************************\
 
Module: Convert goto programs to binary format and back (with irep
        hashing)
 
Author: CM Wintersteiger
 
Date: May 2007
 
\*******************************************************************/

#include <goto-programs/goto_program_irep.h>
#include <goto-programs/goto_program_serialization.h>
#include <sstream>
#include <util/irep_serialization.h>

void goto_program_serializationt::convert(
  const goto_programt &goto_program,
  std::ostream &out)
{
  irepcache.emplace_back();
  ::convert(goto_program, irepcache.back());
  irepconverter.reference_convert(irepcache.back(), out);
}

void goto_program_serializationt::convert(std::istream &in, irept &gprep)
{
  irepconverter.reference_convert(in, gprep);
  // reference is not resolved here!
}
