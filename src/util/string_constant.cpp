/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <arith_tools.h>
#include <std_types.h>
#include <c_types.h>

#include "string_constant.h"

string_constantt::string_constantt():exprt("string-constant")
{
  type()=array_typet(char_type());
  set_value(irep_idt());
}

string_constantt::string_constantt(const irep_idt &_value):
  exprt("string-constant")
{
  type()=array_typet(char_type());
  set_value(_value);
}

void string_constantt::set_value(const irep_idt &value)
{
  exprt size_expr=from_integer(value.size()+1, int_type());
  type().size(size_expr);
  exprt::value(value);
}
