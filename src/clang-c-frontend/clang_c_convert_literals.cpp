/*
 * clang_c_convert_literals.cpp
 *
 *  Created on: Jul 23, 2015
 *      Author: mramalho
 */

#include <clang-c-frontend/clang_c_convert.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/string_constant.h>

bool clang_c_convertert::convert_character_literal(
  const clang::CharacterLiteral &char_literal,
  exprt &dest)
{
  typet type;
  if(get_type(char_literal.getType(), type))
    return true;

  dest =
    constant_exprt(
      integer2binary(char_literal.getValue(), bv_width(type)),
      integer2string(char_literal.getValue()),
      type);

  return false;
}

bool clang_c_convertert::convert_string_literal(
  const clang::StringLiteral &string_literal,
  exprt &dest)
{
  typet type;
  if(get_type(string_literal.getType(), type))
    return true;

  string_constantt string(string_literal.getBytes().str(), type);
  dest.swap(string);

  return false;
}

bool clang_c_convertert::convert_integer_literal(
  const clang::IntegerLiteral &integer_literal,
  exprt &dest)
{
  typet type;
  if(get_type(integer_literal.getType(), type))
    return true;

  assert(type.is_unsignedbv() || type.is_signedbv());

  llvm::APInt val = integer_literal.getValue();

  exprt the_val;
  if (type.is_unsignedbv())
  {
    the_val =
      constant_exprt(
        integer2binary(val.getZExtValue(), bv_width(type)),
        integer2string(val.getZExtValue()),
        type);
  }
  else
  {
    the_val =
      constant_exprt(
        integer2binary(val.getSExtValue(), bv_width(type)),
        integer2string(val.getSExtValue()),
        type);
  }

  dest.swap(the_val);
  return false;
}

static void parse_float(
  llvm::SmallVectorImpl<char> &src,
  mp_integer &significand,
  mp_integer &exponent)
{
  // {digit}{dot}{digits}[+-]{exponent}

  unsigned p = 0;

  // get whole number
  std::string str_whole_number = "";
  str_whole_number += src[p++];

  // skip dot
  assert(src[p] == '.');
  p++;

  // get fraction part
  std::string str_fraction_part = "";
  while (src[p] != 'E')
    str_fraction_part += src[p++];

  // skip E
  assert(src[p] == 'E');
  p++;

  // get exponent
  assert(src[p] == '+' || src[p] == '-');

  // skip +
  if(src[p] == '+')
    p++;

  std::string str_exponent = "";
  str_exponent += src[p++];

  while (p < src.size())
    str_exponent += src[p++];

  std::string str_number = str_whole_number + str_fraction_part;

  if (str_number.empty())
    significand = 0;
  else
    significand = string2integer(str_number);

  if (str_exponent.empty())
    exponent = 0;
  else
    exponent = string2integer(str_exponent);

  exponent -= str_fraction_part.size();
}

bool clang_c_convertert::convert_float_literal(
  const clang::FloatingLiteral &floating_literal,
  exprt &dest)
{
  typet type;
  if(get_type(floating_literal.getType(), type))
    return true;

  // Get the value and convert it to string
  llvm::APFloat val = floating_literal.getValue();

  // We create a vector of char with the maximum size a double can be:
  // long doubles on 64 bits machines can be 128 bits long
  llvm::SmallVector<char, 128> string;

  // However, only get the number up to the width of the number
  // Can be 32 (floats), 64 (doubles) or 128 (long doubles)
  unsigned width = bv_width(type);
  val.toString(string, width, 0);

  // Now let's build the literal
  mp_integer value;
  mp_integer significand;
  mp_integer exponent;
  std::string value_string;

  // Fixed bvs
  if(config.ansi_c.use_fixed_for_float)
  {
    // If it's +oo or -oo, we can't parse it
    if(val.isInfinity())
    {
      if(val.isNegative())
      {
        // saturate: use "smallest value"
        value = -power(2, width - 1);
      }
      else
      {
        // saturate: use "biggest value"
        value = power(2, width - 1) - 1;
      }
    }
    else if(val.isNaN())
    {
      value = 0;
    }
    else
    {
      // Everything else is fine
      parse_float(string, significand, exponent);

      unsigned fraction_bits;
      const std::string &integer_bits = type.integer_bits().as_string();

      if (integer_bits == "")
        fraction_bits = width / 2;
      else
        fraction_bits = width - atoi(integer_bits.c_str());

      mp_integer factor = mp_integer(1) << fraction_bits;
      value = significand * factor;

      if(exponent < 0)
        value /= power(10, -exponent);
      else
      {
        value *= power(10, exponent);

        if(value <= -power(2, width - 1) - 1)
        {
          // saturate: use "smallest value"
          value = -power(2, width - 1);
        }
        else if(value >= power(2, width - 1))
        {
          // saturate: use "biggest value"
          value = power(2, width - 1) - 1;
        }
      }
    }

    // Save value string format
    value_string = integer2string(value);
  }
  else
  {
    ieee_floatt a;
    a.spec=to_floatbv_type(type);

    // If it's +oo or -oo, we can't parse it
    if(val.isInfinity())
    {
      if(val.isNegative()) {
        a = ieee_floatt::minus_infinity(a.spec);
      } else {
        a = ieee_floatt::plus_infinity(a.spec);
      }
    }
    else if(val.isNaN())
    {
      a = ieee_floatt::NaN(a.spec);
    }
    else
    {
      parse_float(string, significand, exponent);

      a.from_base10(significand, exponent);
    }

    // Pack the value to generate the correct number, regardless of the case
    value = a.pack();

    // Save value string format
    value_string = a.to_ansi_c_string();
  }

  dest =
    constant_exprt(
      integer2binary(value, bv_width(type)),
      value_string,
      type);

  return false;
}
