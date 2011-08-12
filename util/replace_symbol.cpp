/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "replace_symbol.h"
#include <std_types.h>

/*******************************************************************\

Function: replace_symbolt::replace_symbolt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

replace_symbolt::replace_symbolt()
{
}

/*******************************************************************\

Function: replace_symbolt::~replace_symbolt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

replace_symbolt::~replace_symbolt()
{
}

/*******************************************************************\

Function: replace_symbolt::replace

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool replace_symbolt::replace(exprt &dest)
{
  if(dest.is_symbol())
  {
    expr_mapt::const_iterator it=
      expr_map.find(dest.identifier());

    if(it!=expr_map.end())
    {
      dest=it->second;
      return false;
    }
  }

  bool result=true;

  Forall_operands(it, dest)
    result=replace(*it) && result;

  result=replace(dest.type()) && result;

  return result;
}

/*******************************************************************\

Function: replace_symbolt::replace

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool replace_symbolt::replace(typet &dest)
{
  if(dest.has_subtype())
    replace(dest.subtype());

  Forall_subtypes(it, dest)
    replace(*it);
    
  if(dest.is_struct() ||
     dest.is_union())
  {
    struct_typet &struct_type = to_struct_type(dest);    
    struct_typet::componentst &components = struct_type.components();
    for (struct_typet::componentst::iterator it = components.begin();
         it!=components.end();
         it++)
      replace(*it);
  } 
  else if(dest.is_code())
  {
    code_typet &code_type=to_code_type(dest);
    code_typet::argumentst &arguments=code_type.arguments();
    for (code_typet::argumentst::iterator it = arguments.begin();
         it!=arguments.end();
         it++)
      replace(*it);
  }
  
  if(dest.is_symbol())
  {
    type_mapt::const_iterator it=
      type_map.find(dest.identifier());

    if(it!=type_map.end())
    {
      dest=it->second;
      return false;
    }
  }

  return true;
}
