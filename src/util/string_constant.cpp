#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/config.h>
#include <util/c_types.h>
#include <util/std_expr.h>
#include <util/std_types.h>
#include <util/string_constant.h>

string_constantt::string_constantt(const irep_idt &value)
  : string_constantt(value, array_typet(char_type()))
{
}

string_constantt::string_constantt(const irep_idt &value, const typet &type)
  : exprt("string-constant", type)
{
  set_value(value);
}

void string_constantt::set_value(const irep_idt &value)
{
  exprt::value(value);
}
