/*******************************************************************\

Module: SpecC Language Conversion

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/ansi_c_convert_type.h>
#include <cassert>
#include <iostream>
#include <util/arith_tools.h>
#include <util/config.h>

void ansi_c_convert_typet::read(const typet &type)
{
  clear();
  location = type.location();
  read_rec(type);
}

void ansi_c_convert_typet::read_rec(const typet &type)
{
  if(type.id() == "merged_type")
  {
    forall_subtypes(it, type)
      read_rec(*it);
  }
  else if(type.id() == "signed")
    signed_cnt++;
  else if(type.id() == "unsigned")
    unsigned_cnt++;
  else if(type.id() == "volatile")
    c_qualifiers.is_volatile = true;
  else if(type.id() == "const")
    c_qualifiers.is_constant = true;
  else if(type.id() == "restricted")
    c_qualifiers.is_restricted = true;
  else if(type.id() == "char")
    char_cnt++;
  else if(type.id() == "int")
    int_cnt++;
  else if(type.id() == "int8")
    int8_cnt++;
  else if(type.id() == "int16")
    int16_cnt++;
  else if(type.id() == "int32")
    int32_cnt++;
  else if(type.id() == "int64")
    int64_cnt++;
  else if(type.id() == "ptr32")
    ptr32_cnt++;
  else if(type.id() == "ptr64")
    ptr64_cnt++;
  else if(type.id() == "short")
    short_cnt++;
  else if(type.id() == "long")
    long_cnt++;
  else if(type.id() == "double")
    double_cnt++;
  else if(type.id() == "float")
    float_cnt++;
  else if(type.is_bool())
    bool_cnt++;
  else if(type.id() == "static")
    c_storage_spec.is_static = true;
  else if(type.id() == "inline")
    c_storage_spec.is_inline = true;
  else if(type.id() == "extern")
    c_storage_spec.is_extern = true;
  else if(type.id() == "typedef")
    c_storage_spec.is_typedef = true;
  else if(type.id() == "register")
    c_storage_spec.is_register = true;
  else if(type.id() == "auto")
  {
    // ignore
  }
  else if(type == get_nil_irep())
  {
    // ignore
  }
  else
    other.push_back(type);
}

void ansi_c_convert_typet::write(typet &type)
{
  type.clear();

  // first, do "other"

  if(!other.empty())
  {
    if(
      double_cnt || float_cnt || signed_cnt || unsigned_cnt || int_cnt ||
      bool_cnt || short_cnt || char_cnt || int8_cnt || int16_cnt || int32_cnt ||
      int64_cnt || ptr32_cnt || ptr64_cnt || long_cnt)
    {
      err_location(location);
      error("illegal type modifier for defined type");
      throw 0;
    }

    if(other.size() != 1)
    {
      err_location(location);
      error("illegal combination of defined types");
      throw 0;
    }

    type.swap(other.front());
  }
  else if(double_cnt || float_cnt)
  {
    if(
      signed_cnt || unsigned_cnt || int_cnt || bool_cnt || int8_cnt ||
      int16_cnt || int32_cnt || int64_cnt || ptr32_cnt || ptr64_cnt ||
      short_cnt || char_cnt)
    {
      err_location(location);
      error("cannot conbine integer type with float");
      throw 0;
    }

    if(double_cnt && float_cnt)
    {
      err_location(location);
      error("conflicting type modifiers");
      throw 0;
    }

    if(long_cnt == 0)
    {
      if(double_cnt != 0)
        type = double_type();
      else
        type = float_type();
    }
    else if(long_cnt == 1 || long_cnt == 2)
    {
      if(double_cnt != 0)
        type = long_double_type();
      else
      {
        err_location(location);
        error("conflicting type modifiers");
        throw 0;
      }
    }
    else
    {
      err_location(location);
      error("illegal type modifier for float");
      throw 0;
    }
  }
  else if(bool_cnt)
  {
    if(
      signed_cnt || unsigned_cnt || int_cnt || short_cnt || int8_cnt ||
      int16_cnt || int32_cnt || int64_cnt || ptr32_cnt || ptr64_cnt ||
      char_cnt || long_cnt)
    {
      err_location(location);
      error("illegal type modifier for boolean type");
      throw 0;
    }

    type.id("bool");
  }
  else if(ptr32_cnt || ptr64_cnt)
  {
    type.id("pointer");
    type.subtype() = typet("empty");
  }
  else
  {
    // it is integer -- signed or unsigned?

    if(signed_cnt && unsigned_cnt)
    {
      err_location(location);
      error("conflicting type modifiers");
      throw 0;
    }
    if(unsigned_cnt)
      type.id("unsignedbv");
    else if(signed_cnt)
      type.id("signedbv");
    else
    {
      if(char_cnt)
        type.id(config.ansi_c.char_is_unsigned ? "unsignedbv" : "signedbv");
      else
        type.id("signedbv");
    }

    // get width

    unsigned width;

    if(int8_cnt || int16_cnt || int32_cnt || int64_cnt)
    {
      if(long_cnt || char_cnt || short_cnt)
      {
        err_location(location);
        error("conflicting type modifiers");
        throw 0;
      }

      if(int8_cnt)
        width = 1 * 8;
      else if(int16_cnt)
        width = 2 * 8;
      else if(int32_cnt)
        width = 4 * 8;
      else if(int64_cnt)
        width = 8 * 8;
      else
        abort();
    }
    else if(short_cnt)
    {
      if(long_cnt || char_cnt)
      {
        err_location(location);
        error("conflicting type modifiers");
        throw 0;
      }

      width = config.ansi_c.short_int_width;
    }
    else if(char_cnt)
    {
      if(long_cnt)
      {
        err_location(location);
        error("illegal type modifier for char type");
        throw 0;
      }

      width = config.ansi_c.char_width;
    }
    else if(long_cnt == 0)
    {
      width = config.ansi_c.int_width;
    }
    else if(long_cnt == 1)
    {
      width = config.ansi_c.long_int_width;
    }
    else if(long_cnt == 2)
    {
      width = config.ansi_c.long_long_int_width;
    }
    else
    {
      err_location(location);
      error("illegal type modifier for integer type");
      throw 0;
    }

    type.width(width);
  }

  c_qualifiers.write(type);
}
