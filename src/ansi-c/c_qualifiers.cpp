/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "c_qualifiers.h"

/*******************************************************************\

Function: c_qualifierst::as_string

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string c_qualifierst::as_string() const
{
  std::string qualifiers;
  
  if(is_constant)
    qualifiers+="const ";

  if(is_volatile)
    qualifiers+="volatile ";

  if(is_restricted)
    qualifiers+="restricted ";
    
  return qualifiers;
}

/*******************************************************************\

Function: c_qualifierst::read

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_qualifierst::read(const typet &src)
{
  if(src.cmt_constant())
    is_constant=true;

  if(src.cmt_volatile())
    is_volatile=true;

  if(src.restricted())
    is_restricted=true;
}

/*******************************************************************\

Function: c_qualifierst::write

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_qualifierst::write(typet &dest) const
{
  if(is_constant)
    dest.cmt_constant(true);
  else
    dest.remove("#constant");

  if(is_volatile)
    dest.cmt_volatile(true);
  else
    dest.remove("#volatile");

  if(is_restricted)
    dest.restricted(true);
  else
    dest.remove("#restricted");
}

/*******************************************************************\

Function: c_qualifierst::clear

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_qualifierst::clear(typet &dest)
{
  dest.remove("#constant");
  dest.remove("#volatile");
  dest.remove("#restricted");
}

