/*
 * llvmtypecheck.cpp
 *
 *  Created on: Jul 23, 2015
 *      Author: mramalho
 */

#include "llvm_convert.h"

#include <arith_tools.h>
#include <bitvector.h>
#include <expr_util.h>

#include <ansi-c/c_types.h>
#include <ansi-c/ansi_c_expr.h>

#include <sstream>

void llvm_convertert::convert_character_literal(
  const clang::CharacterLiteral char_literal,
  exprt &dest)
{
  typet type;
  get_type(char_literal.getType(), type);

  dest =
    constant_exprt(
      integer2binary(char_literal.getValue(), bv_width(type)),
      integer2string(char_literal.getValue()),
      type);
}

void llvm_convertert::convert_string_literal(
  const clang::StringLiteral string_literal,
  exprt &dest)
{
  typet type;
  get_type(string_literal.getType(), type);

  string_constantt string;
  string.set_value(string_literal.getString().str());

  dest.swap(string);
}

void llvm_convertert::convert_integer_literal(
  llvm::APInt val,
  typet type,
  exprt &dest)
{
  assert(type.is_unsignedbv() || type.is_signedbv());

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
}

void llvm_convertert::convert_float_literal(
  llvm::APFloat val,
  typet type,
  exprt &dest)
{
  llvm::SmallVector<char, 16> string;
  val.toString(string, 32, 0);

  mp_integer significand;
  mp_integer exponent;

  parse_float(string, significand, exponent);

  dest = constant_exprt(type);

  if(config.ansi_c.use_fixed_for_float)
  {
    unsigned width = atoi(dest.type().width().c_str());
    unsigned fraction_bits;
    const std::string &integer_bits = dest.type().integer_bits().as_string();

    if (integer_bits == "")
      fraction_bits = width / 2;
    else
      fraction_bits = width - atoi(integer_bits.c_str());

    mp_integer factor = mp_integer(1) << fraction_bits;
    mp_integer value = significand * factor;

    if (exponent < 0)
      value /= power(10, -exponent);
    else
      value *= power(10, exponent);

    dest.value(integer2binary(value, width));
  }
  else
  {
    std::cerr << "floatbv unsupported, sorry" << std::endl;
    abort();
  }
}

void llvm_convertert::parse_float(
  llvm::SmallVector<char, 16> &src,
  mp_integer &significand,
  mp_integer &exponent)
{
  // {digit}{dot}{31 digits}[+-]{exponent}

  unsigned p = 0;

  // get whole number
  std::string str_whole_number = "";
  str_whole_number += src[p++];

  // skip dot
  assert(src[p++] == '.');

  // get fraction part
  std::string str_fraction_part = "";
  while (src[p] != 'E')
    str_fraction_part += src[p++];

  // skip E
  assert(src[p++] == 'E');

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
