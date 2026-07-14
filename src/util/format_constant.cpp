#include <util/fixedbv.h>
#include <util/format_constant.h>
#include <util/ieee_float.h>
#include <irep2/irep2_utils.h>
#include <util/mp_arith.h>

std::string format_constantt::operator()(const expr2tc &expr)
{
  if (is_constant_expr(expr))
  {
    // Format the full-width value directly from the BigInt. Going through
    // as_ulong()/as_long() would silently truncate any constant wider than
    // 64 bits (e.g. __int128) to its low 64 bits (finding R2); integer2string
    // prints the exact value at any width and handles the sign.
    if (is_unsignedbv_type(expr) || is_signedbv_type(expr))
      return integer2string(to_constant_int2t(expr).value);

    if (is_fixedbv_type(expr))
      return fixedbvt(to_constant_fixedbv2t(expr).value).format(*this);

    if (is_floatbv_type(expr))
      return ieee_floatt(to_constant_floatbv2t(expr).value).format(*this);

    if (is_constant_string2t(expr))
      return to_constant_string2t(expr).value.as_string();
  }

  return "(format-constant failed: " + get_expr_id(expr) + ")";
}
