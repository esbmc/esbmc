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
  /* Fails for L"" and other large character types, because the below
   * computation is buggy for those. See also #1165. */
  assert(bv_width(type().subtype()) == config.ansi_c.char_width);

  exprt size_expr = constant_exprt(
    integer2binary(value.size() + 1, bv_width(size_type())),
    integer2string(value.size() + 1),
    size_type());
  type().size(size_expr);
  exprt::value(value);
}
