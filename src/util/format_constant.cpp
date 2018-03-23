/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/fixedbv.h>
#include <util/format_constant.h>
#include <util/ieee_float.h>
#include <util/irep2_utils.h>
#include <util/i2string.h>

std::string format_constantt::operator()(const expr2tc &expr)
{
  if(is_constant_expr(expr))
  {
    if(is_unsignedbv_type(expr))
      return i2string(to_constant_int2t(expr).as_ulong());

    if(is_signedbv_type(expr))
      return i2string(to_constant_int2t(expr).as_long());

    if(is_fixedbv_type(expr))
      return fixedbvt(to_constant_fixedbv2t(expr).value).format(*this);

    if(is_floatbv_type(expr))
      return ieee_floatt(to_constant_floatbv2t(expr).value).format(*this);

    if(is_string_type(expr))
      return to_constant_string2t(expr).value.as_string();
  }

  return "(format-constant failed: " + get_expr_id(expr) + ")";
}
