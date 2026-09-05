#include <python-frontend/math/convert_float_literal.h>
#include <python-frontend/math/parse_float.h>
#include <python-frontend/type/type_utils.h>
#include <util/arith/arith_tools.h>
#include <util/lang/c_types.h>
#include <util/config/config.h>
#include <util/arith/ieee_float.h>
#include <util/irep/std_types.h>

void convert_float_literal(const std::string &src, exprt &dest)
{
  BigInt significand;
  BigInt exponent;
  bool is_float, is_long;
  unsigned base;

  parse_float(src, significand, exponent, base, is_float, is_long);

  dest = exprt("constant");

  dest.cformat(src);

  if (is_float)
  {
    dest.type() = float_type();
    type_utils::set_cpp_type(dest.type(), "float");
  }
  else if (is_long)
  {
    dest.type() = long_double_type();
    type_utils::set_cpp_type(dest.type(), "long_double");
  }
  else
  {
    dest.type() = double_type();
    type_utils::set_cpp_type(dest.type(), "double");
  }

  ieee_floatt a;

  a.spec = to_floatbv_type(dest.type());

  if (base == 10)
    a.from_base10(significand, exponent);
  else if (base == 2) // hex
    a.build(significand, exponent);
  else
    assert(false);

  dest.value(integer2binary(a.pack(), a.spec.width()));
}
