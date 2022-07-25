#include <util/c_types.h>
#include <util/config.h>
#include <util/std_types.h>
#include <irep2/irep2_utils.h>

typet build_float_type(unsigned width)
{
  if(config.ansi_c.use_fixed_for_float)
  {
    fixedbv_typet result;
    result.set_width(width);
    result.set_integer_bits(width / 2);
    return std::move(result);
  }
  floatbv_typet result;
  result.set_width(width);

  switch(width)
  {
  case 16:
    result.set_f(11);
    break;
  case 32:
    result.set_f(23);
    break;
  case 64:
    result.set_f(52);
    break;
  case 96:
    result.set_f(80);
    break;
  case 128:
    result.set_f(112);
    break;
  default:
    assert(false);
  }

  return std::move(result);
}

type2tc build_float_type2(unsigned width)
{
  if(config.ansi_c.use_fixed_for_float)
  {
    fixedbv_type2tc result(width, width / 2);
    return result;
  }

  unsigned fraction = 0;
  switch(width)
  {
  case 16:
    fraction = 11;
    break;
  case 32:
    fraction = 23;
    break;
  case 64:
    fraction = 52;
    break;
  case 96:
    fraction = 80;
    break;
  case 128:
    fraction = 112;
    break;
  default:
    assert(false);
  }

  floatbv_type2tc result(fraction, width - fraction - 1);
  return result;
}

typet index_type()
{
  return signed_size_type();
}

type2tc index_type2()
{
  return get_int_type(config.ansi_c.int_width);
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
  return get_int_type(config.ansi_c.int_width);
}

typet uint_type()
{
  return unsignedbv_typet(config.ansi_c.int_width);
}

type2tc uint_type2()
{
  return get_uint_type(config.ansi_c.int_width);
}

typet bool_type()
{
  typet result = bool_typet();
  return result;
}

typet long_int_type()
{
  return signedbv_typet(config.ansi_c.long_int_width);
}

type2tc long_int_type2()
{
  return get_int_type(config.ansi_c.long_int_width);
}

typet long_long_int_type()
{
  return signedbv_typet(config.ansi_c.long_long_int_width);
}

type2tc long_long_int_type2()
{
  return get_int_type(config.ansi_c.long_long_int_width);
}

typet int128_type()
{
  return signedbv_typet(128);
}

type2tc int128_type2()
{
  return get_int_type(128);
}

typet uint128_type()
{
  return unsignedbv_typet(128);
}

type2tc uint128_type2()
{
  return get_uint_type(128);
}

typet long_uint_type()
{
  return unsignedbv_typet(config.ansi_c.long_int_width);
}

type2tc long_uint_type2()
{
  return get_uint_type(config.ansi_c.long_int_width);
}

typet long_long_uint_type()
{
  return unsignedbv_typet(config.ansi_c.long_long_int_width);
}

type2tc long_long_uint_type2()
{
  return get_uint_type(config.ansi_c.long_long_int_width);
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
  if(config.ansi_c.char_is_unsigned)
    return get_uint_type(config.ansi_c.char_width);
  return get_int_type(config.ansi_c.char_width);
}

typet half_float_type()
{
  return build_float_type(config.ansi_c.short_int_width);
}

typet float_type()
{
  return build_float_type(config.ansi_c.single_width);
}

type2tc float_type2()
{
  return build_float_type2(config.ansi_c.single_width);
}

typet double_type()
{
  return build_float_type(config.ansi_c.double_width);
}

type2tc double_type2()
{
  return build_float_type2(config.ansi_c.double_width);
}

typet long_double_type()
{
  return build_float_type(config.ansi_c.long_double_width);
}

type2tc long_double_type2()
{
  return build_float_type2(config.ansi_c.long_double_width);
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

type2tc get_uint8_type()
{
  static type2tc ubv8(new unsignedbv_type2t(8));
  return ubv8;
}

type2tc get_uint16_type()
{
  static type2tc ubv16(new unsignedbv_type2t(16));
  return ubv16;
}

type2tc get_uint32_type()
{
  static type2tc ubv32(new unsignedbv_type2t(32));
  return ubv32;
}

type2tc get_uint64_type()
{
  static type2tc ubv64(new unsignedbv_type2t(64));
  return ubv64;
}

type2tc get_int8_type()
{
  static type2tc sbv8(new signedbv_type2t(8));
  return sbv8;
}

type2tc get_int16_type()
{
  static type2tc sbv16(new signedbv_type2t(16));
  return sbv16;
}

type2tc get_int32_type()
{
  static type2tc sbv32(new signedbv_type2t(32));
  return sbv32;
}

type2tc get_int64_type()
{
  static type2tc sbv64(new signedbv_type2t(64));
  return sbv64;
}

type2tc get_uint_type(unsigned int sz)
{
  switch(sz)
  {
  case 8:
    return get_uint8_type();
  case 16:
    return get_uint16_type();
  case 32:
    return get_uint32_type();
  case 64:
    return get_uint64_type();
  default:;
  }
  return type2tc(new unsignedbv_type2t(sz));
}

type2tc get_int_type(unsigned int sz)
{
  switch(sz)
  {
  case 8:
    return get_int8_type();
  case 16:
    return get_int16_type();
  case 32:
    return get_int32_type();
  case 64:
    return get_int64_type();
  default:;
  }
  return type2tc(new signedbv_type2t(sz));
}

type2tc get_bool_type()
{
  static type2tc bool_type = type2tc(new bool_type2t());
  return bool_type;
}

type2tc get_empty_type()
{
  static type2tc empty_type = type2tc(new empty_type2t());
  return empty_type;
}
