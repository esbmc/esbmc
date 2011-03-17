/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "promela_parser.h"

promela_parsert promela_parser;

/*******************************************************************\

Function: promela_parsert::lookup

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

#if 0
promela_id_classt promela_parsert::lookup(std::string &name) const
{
  for(scopest::const_reverse_iterator it=scopes.rbegin();
      it!=scopes.rend(); it++)
  {
    scopet::name_mapt::const_iterator n_it=it->name_map.find(name);
    if(n_it!=it->name_map.end())
    {
      name=it->prefix+name;
      return n_it->second.id_class;
    }
  }

  return SC_UNKNOWN;
}
#endif
