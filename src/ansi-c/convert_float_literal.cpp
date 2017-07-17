/*******************************************************************\

Module: C++ Language Conversion

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/convert_float_literal.h>
#include <ansi-c/parse_float.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/ieee_float.h>
#include <util/std_types.h>

void convert_float_literal(
  const std::string &src,
  exprt &dest)
{
  mp_integer significand;
  mp_integer exponent;
  bool is_float, is_long;
  unsigned base;

  parse_float(src, significand, exponent, base, is_float, is_long);

  dest=exprt("constant");

  dest.cformat(src);

  if(is_float) {
    dest.type()=float_type();
    dest.type().set("#cpp_type", "float");
  } else if(is_long) {
    dest.type()=long_double_type();
    dest.type().set("#cpp_type", "long_double");
  } else {
    dest.type()=double_type();
    dest.type().set("#cpp_type", "double");
  }

  if(config.ansi_c.use_fixed_for_float)
  {
    unsigned width=atoi(dest.type().width().c_str());
    unsigned fraction_bits;
    const std::string &integer_bits=dest.type().integer_bits().as_string();

    if(integer_bits=="")
      fraction_bits=width/2;
    else
      fraction_bits=width-atoi(integer_bits.c_str());

    mp_integer factor=mp_integer(1)<<fraction_bits;
    mp_integer value=significand*factor;

    if(value!=0)
    {
      if(exponent<0)
        value/=power(base, -exponent);
      else
      {
        value*=power(base, exponent);

        if(value>=power(2, width-1))
        {
          // saturate: use "biggest value"
          value=power(2, width-1)-1;
        }
        else if(value<=-power(2, width-1)-1)
        {
          // saturate: use "smallest value"
          value=-power(2, width-1);
        }
      }
    }

    dest.value(integer2binary(value, width));
  }
  else
  {
    ieee_floatt a;

    a.spec=to_floatbv_type(dest.type());

    if(base==10)
      a.from_base10(significand, exponent);
    else if(base==2) // hex
      a.build(significand, exponent);
    else
      assert(false);

    dest.value(integer2binary(a.pack(), a.spec.width()));
  }
}
