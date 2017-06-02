/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/c_types.h>
#include <util/config.h>
#include <util/std_types.h>
#include <util/irep2_type.h>

typet build_float_type(unsigned width)
{
  if(config.ansi_c.use_fixed_for_float)
  {
    fixedbv_typet result;
    result.set_width(width);
    result.set_integer_bits(width/2);
    return result;
  }
  else
  {
    floatbv_typet result;
    result.set_width(width);

    switch(width)
    {
    case 32: result.set_f(23); break;
    case 64: result.set_f(52); break;
    case 128: result.set_f(112); break;
    default: assert(false);
    }

    return result;
  }
}

typet index_type()
{
  return signedbv_typet(config.ansi_c.int_width);
}

type2tc index_type2(void)
{
  return type_pool.get_int(config.ansi_c.int_width);
}

typet enum_type()
{
  return signedbv_typet(config.ansi_c.int_width);
}

typet int_type()
{
  return signedbv_typet(config.ansi_c.int_width);
}

type2tc int_type2()
{
  return type_pool.get_int(config.ansi_c.int_width);
}

typet uint_type()
{
  return unsignedbv_typet(config.ansi_c.int_width);
}

type2tc uint_type2()
{
  return type_pool.get_uint(config.ansi_c.int_width);
}

typet bool_type()
{
  typet result=bool_typet();
  return result;
}

typet long_int_type()
{
  return signedbv_typet(config.ansi_c.long_int_width);
}

typet long_long_int_type()
{
  return signedbv_typet(config.ansi_c.long_long_int_width);
}

typet long_uint_type()
{
  return unsignedbv_typet(config.ansi_c.long_int_width);
}

typet long_long_uint_type()
{
  return unsignedbv_typet(config.ansi_c.long_long_int_width);
}

typet signed_short_int_type()
{
  return signedbv_typet(config.ansi_c.short_int_width);
}

typet unsigned_short_int_type()
{
  return unsignedbv_typet(config.ansi_c.short_int_width);
}

typet char_type()
{
  if(config.ansi_c.char_is_unsigned)
    return unsignedbv_typet(config.ansi_c.char_width);

  return signedbv_typet(config.ansi_c.char_width);
}

typet unsigned_char_type()
{
  return unsignedbv_typet(config.ansi_c.char_width);
}

typet signed_char_type()
{
  return signedbv_typet(config.ansi_c.char_width);
}

typet char16_type()
{
  return unsignedbv_typet(config.ansi_c.short_int_width);
}

typet char32_type()
{
  return unsignedbv_typet(config.ansi_c.single_width);
}

typet wchar_type()
{
  return signedbv_typet(config.ansi_c.int_width);
}

typet unsigned_wchar_type()
{
  return unsignedbv_typet(config.ansi_c.int_width);
}

type2tc char_type2()
{
  if (config.ansi_c.char_is_unsigned)
    return type_pool.get_uint(config.ansi_c.char_width);
  else
    return type_pool.get_int(config.ansi_c.char_width);
}

typet float_type()
{
  return build_float_type(config.ansi_c.single_width);
}

typet double_type()
{
  return build_float_type(config.ansi_c.double_width);
}

typet long_double_type()
{
  return build_float_type(config.ansi_c.long_double_width);
}

typet size_type()
{
  return unsignedbv_typet(config.ansi_c.pointer_width);
}

typet signed_size_type()
{
  return signedbv_typet(config.ansi_c.pointer_width);
}

typet pointer_type()
{
  return unsignedbv_typet(config.ansi_c.pointer_width);
}

type2tc pointer_type2()
{
  return type2tc(new unsignedbv_type2t(config.ansi_c.pointer_width));
}
