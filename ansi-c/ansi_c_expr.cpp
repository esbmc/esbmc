/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <arith_tools.h>

#include "ansi_c_expr.h"
#include "c_types.h"

/*******************************************************************\

Function: string_constantt::set_value

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

string_constantt::string_constantt():exprt("string-constant")
{
  set_value("");
  type()=typet("array");
  type().subtype()=char_type();
}

/*******************************************************************\

Function: string_constantt::set_value

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_constantt::set_value(const irep_idt &value)
{
  exprt size_expr=from_integer(value.size()+1, int_type());
  type().size(size_expr);
  exprt::value(value);
}
