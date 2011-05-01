/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "type.h"

/*******************************************************************\

Function: typet::copy_to_subtypes

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void typet::copy_to_subtypes(const typet &type)
{
  subtypes().push_back(type);
}

/*******************************************************************\

Function: typet::move_to_subtypes

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void typet::move_to_subtypes(typet &type)
{
  subtypest &sub=subtypes();
  sub.push_back(static_cast<const typet &>(get_nil_irep()));
  sub.back().swap(type);
}

/*******************************************************************\

Function: is_number

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool is_number(const typet &type)
{
  const std::string &id=type.id_string();
  return id=="rational" ||
         id=="real" ||
         id=="integer" ||
         id=="natural" || 
         id=="complex" ||
         id=="unsignedbv" ||
         id=="signedbv" || 
         id=="floatbv" ||
         id=="fixedbv";
}

irep_idt typet::f_subtype = dstring("subtype");
irep_idt typet::f_subtypes = dstring("subtypes");
irep_idt typet::f_location = dstring("#location");
