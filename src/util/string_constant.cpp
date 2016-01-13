/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <arith_tools.h>
#include <std_types.h>
#include <c_types.h>

#include "string_constant.h"

string_constantt::string_constantt()
 : string_constantt(irep_idt(), char_type())
{
}

string_constantt::string_constantt(const irep_idt &value)
  : string_constantt(value, char_type())
{
}

string_constantt::string_constantt(const irep_idt &value, const typet type)
  : exprt("string-constant", array_typet(type))
{
  set_value(value);
}

void string_constantt::set_value(const irep_idt &value)
{
  exprt size_expr = from_integer(value.size() + 1, uint_type());
  type().size(size_expr);
  exprt::value(value);
}
