#include <util/compiler_defs.h>
// Remove warnings from Clang headers
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <clang/AST/Expr.h>
CC_DIAGNOSTIC_POP()

#include <clang-c-frontend/clang_c_convert.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/ieee_float.h>
#include <util/string_constant.h>

bool clang_c_convertert::convert_character_literal(
  const clang::CharacterLiteral &char_literal,
  exprt &dest)
{
  typet type;
  if (get_type(char_literal.getType(), type))
    return true;

  dest = constant_exprt(
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
  if (get_type(string_literal.getType(), type))
    return true;

  // When strings are just used for initialization, it might be worth
  // to just replace them by null. See https://github.com/esbmc/esbmc/pull/1185#issuecomment-1631166527
  if (config.options.get_bool_option("no-string-literal"))
  {
    auto value = gen_zero(type);
    dest.swap(value);
  }
  else
  {
    irep_idt kind = string_constantt::k_default;
    if (string_literal.isWide())
      kind = string_constantt::k_wide;
    else if (
      string_literal.isUTF8() || string_literal.isUTF16() ||
      string_literal.isUTF32())
      kind = string_constantt::k_unicode;
    string_constantt string(string_literal.getBytes().str(), type, kind);
    dest.swap(string);
  }

  return false;
}

bool clang_c_convertert::convert_integer_literal(
  const clang::IntegerLiteral &integer_literal,
  exprt &dest)
{
  typet type;
  if (get_type(integer_literal.getType(), type))
    return true;

  assert(type.is_unsignedbv() || type.is_signedbv());

  llvm::APInt val = integer_literal.getValue();

  exprt the_val;
  if (type.is_unsignedbv())
  {
    the_val = constant_exprt(
      integer2binary(val.getZExtValue(), bv_width(type)),
      integer2string(val.getZExtValue()),
      type);
  }
  else
  {
    the_val = constant_exprt(
      integer2binary(val.getSExtValue(), bv_width(type)),
      integer2string(val.getSExtValue()),
      type);
  }

  dest.swap(the_val);
  return false;
}

bool clang_c_convertert::convert_float_literal(
  const clang::FloatingLiteral &floating_literal,
  exprt &dest)
{
  typet type;
  if (get_type(floating_literal.getType(), type))
    return true;

  llvm::APFloat val = floating_literal.getValue();
  unsigned width = bv_width(type);
  BigInt value;
  std::string value_string;

  // Fixed-point representation: floats are stored as scaled integers.
  // value = trunc(val * 2^fraction_bits), saturating on overflow.
  if (config.ansi_c.use_fixed_for_float)
  {
    if (val.isInfinity())
    {
      value =
        val.isNegative() ? -power(2, width - 1) : power(2, width - 1) - 1;
    }
    else if (val.isNaN())
    {
      value = 0;
    }
    else
    {
      const std::string &integer_bits = type.integer_bits().as_string();
      unsigned fraction_bits =
        integer_bits.empty() ? width / 2
                             : width - atoi(integer_bits.c_str());

      // scalbn is exact for power-of-2 factors in binary floating-point.
      llvm::APFloat scaled = llvm::scalbn(
        val, (int)fraction_bits, llvm::APFloat::rmNearestTiesToEven);

      llvm::APSInt result(width, /*isUnsigned=*/false);
      bool isExact;
      auto status =
        scaled.convertToInteger(result, llvm::APFloat::rmTowardZero, &isExact);

      // opInvalidOp means the scaled value overflowed the integer width.
      if (status == llvm::APFloat::opInvalidOp)
        value =
          val.isNegative() ? -power(2, width - 1) : power(2, width - 1) - 1;
      else
      {
        llvm::SmallString<64> dec_str;
        result.toString(dec_str, 10, true);
        value = string2integer(dec_str.str().str());
      }
    }

    value_string = integer2string(value);
  }
  else
  {
    // IEEE 754: transfer the exact bit pattern via bitcast to avoid any
    // intermediate decimal rounding.
    ieee_floatt a;
    a.spec = to_floatbv_type(type);

    if (val.isNaN())
      a = ieee_floatt::NaN(a.spec);
    else if (val.isInfinity())
    {
      if (val.isNegative())
        a = ieee_floatt::minus_infinity(a.spec);
      else
        a = ieee_floatt::plus_infinity(a.spec);
    }
    else
    {
      llvm::SmallString<32> hex_str;
      val.bitcastToAPInt().toString(hex_str, 16, false);
      a.unpack(string2integer(hex_str.str().str(), 16));
    }

    value = a.pack();
    value_string = a.to_ansi_c_string();
  }

  dest = constant_exprt(integer2binary(value, width), value_string, type);
  return false;
}
