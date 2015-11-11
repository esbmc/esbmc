/*******************************************************************\
 
Module: Symbol to binary conversions with irep hashing
 
Author: CM Wintersteiger
 
Date: May 2007
 
\*******************************************************************/

#include "symbol_serialization.h"
#include "irep_serialization.h"

void symbol_serializationt::convert(const symbolt& sym, std::ostream &out)
{
  irepcache.push_back(irept());
  sym.to_irep(irepcache.back());  
  irepconverter.reference_convert(irepcache.back(), out);
}

void symbol_serializationt::convert(std::istream& in, irept& symrep)
{  
  irepconverter.reference_convert(in, symrep);
  // reference is not resolved here! 
}
