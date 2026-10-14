/// \file solidity_convert_literals.cpp
/// \brief Literal conversion for the Solidity frontend.
///
/// Converts Solidity literal values (integer literals, boolean literals,
/// string literals, address literals, and rational/fixed-point number
/// literals) from the solc JSON AST into ESBMC's constant expression
/// representation with appropriate typing.

#include <solidity-frontend/solidity_convert.h>
#include <util/arith/arith_tools.h>
#include <util/arith/bitvector.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <util/arith/ieee_float.h>
#include <util/expr/string_constant.h>
#include <util/irep/std_expr.h>

// Integer literal
bool solidity_convertert::convert_integer_literal(
  const nlohmann::json &integer_literal_type,
  std::string the_value,
  exprt &dest)
{
  typet type;
  if (get_type_description(integer_literal_type, type))
    return true;

  the_value.erase(
    std::remove(the_value.begin(), the_value.end(), '_'), the_value.end());

  // Handle decimal-with-exponent literals like `2.5e1`. Move the decimal
  // point right by `exponent` digits when the result lands on an integer
  // boundary; otherwise leave the value untouched and fall through to the
  // generic parser, which will reject the dot. Same idea covers `0.0e0`
  // and similar Solidity rationals that solc has already pre-evaluated to
  // an integer in the typeIdentifier (`t_rational_..._by_1`).
  {
    std::size_t e_pos_d = the_value.find_first_of("eE");
    std::size_t dot = the_value.find('.');
    if (dot != std::string::npos)
    {
      long exp_val = 0;
      if (e_pos_d != std::string::npos)
      {
        try
        {
          exp_val = std::stol(the_value.substr(e_pos_d + 1));
        }
        catch (...)
        {
          return true;
        }
      }
      std::string mantissa =
        (e_pos_d == std::string::npos)
          ? the_value
          : the_value.substr(0, e_pos_d);
      bool neg = !mantissa.empty() && mantissa[0] == '-';
      if (neg)
        mantissa.erase(mantissa.begin());
      std::size_t mdot = mantissa.find('.');
      std::string int_part = mantissa.substr(0, mdot);
      std::string frac_part = mantissa.substr(mdot + 1);
      if (exp_val >= (long)frac_part.size())
      {
        // shift dot right `exp_val` places, padding with zeros
        std::string shifted = int_part + frac_part;
        long pad = exp_val - (long)frac_part.size();
        if (pad > 0)
          shifted += std::string(pad, '0');
        // strip leading zeros (keep at least one)
        std::size_t nz = shifted.find_first_not_of('0');
        if (nz == std::string::npos)
          shifted = "0";
        else
          shifted = shifted.substr(nz);
        the_value = (neg ? "-" : "") + shifted;
      }
      // else: fractional remainder cannot be represented as an integer
      // — bail and let the dot-check below reject it.
    }
  }

  if (the_value.find(".") != std::string::npos)
    return true;

  // Handle scientific notation, e.g., "1e2" -> "100", "200e-2" -> "2".
  // Negative exponents are only legal here when the source actually
  // collapses to an integer (e.g. solc has already reduced the rational
  // to `t_rational_X_by_1`). For an integer mantissa M and exponent -k,
  // M / 10^k is exact iff M % 10^k == 0; otherwise we cannot represent
  // it as an integer and bail out to the generic int parser, which will
  // reject the malformed value.
  std::size_t e_pos = the_value.find_first_of("eE");
  if (e_pos != std::string::npos)
  {
    std::string base_part = the_value.substr(0, e_pos);
    std::string exp_part = the_value.substr(e_pos + 1);

    BigInt base = string2integer(base_part);
    BigInt exponent = string2integer(exp_part);

    if (exponent >= 0)
    {
      BigInt scale = ::power(BigInt(10), exponent);
      BigInt result = base * scale;
      the_value = integer2string(result);
    }
    else
    {
      BigInt scale = ::power(BigInt(10), -exponent);
      // exact integer division only — `power(...)` loops forever on a
      // negative exponent, so we mustn't fall back to it. If the
      // mantissa doesn't divide evenly, give up and let the caller
      // surface a conversion error.
      if (scale == 0 || (base % scale) != 0)
        return true;
      BigInt result = base / scale;
      the_value = integer2string(result);
    }
  }

  exprt the_val;
  // Extract the integer value
  BigInt z_ext_value = string2integer(the_value);

  the_val = constant_exprt(
    integer2binary(z_ext_value, bv_width(type)),
    integer2string(z_ext_value),
    type);

  dest.swap(the_val);
  return false;
}

bool solidity_convertert::convert_bool_literal(
  const nlohmann::json &bool_literal,
  std::string the_value,
  exprt &dest)
{
  typet type;
  if (get_type_description(bool_literal, type))
    return true;

  assert(type.is_bool());

  if (the_value == "true")
  {
    dest = true_exprt();
    return false;
  }

  if (the_value == "false")
  {
    dest = false_exprt();
    return false;
  }

  return true;
}

// TODO: Character literal
/**
 * @brief Converts the string literal to a string_constt
 *
 * @param the_value the value of the literal
 * @param dest return reference
 * @return true Only if the function fails
 * @return false Only if the function successfully converts the literal
 */
bool solidity_convertert::convert_string_literal(
  std::string the_value,
  exprt &dest)
{
  /*
  string-constant
  * type: array
      * size: constant
          * type: unsignedbv
              * width: 64
          * value:
  0000000000000000000000000000000000000000000000000000000000000101
          * #cformat: 5
      * subtype: signedbv
          * width: 8
          * #cpp_type: signed_char
      * #sol_type: STRING_LITERAL
  * value: 1234
  * kind: default
  */
  // TODO: Handle null terminator byte
  string_constantt string(the_value);

  dest.swap(string);
  set_sol_type(dest.type(), SolidityGrammar::SolType::STRING_LITERAL);
  return false;
}

/**
 * convert hex-string to uint constant
 * @n: the bit width, default 256 (unsignedbv_typet(256))
 */
bool solidity_convertert::convert_hex_literal(
  std::string the_value,
  exprt &dest,
  const int n)
{
  // remove "0x" prefix
  if (the_value.length() >= 2)
    if (the_value.substr(0, 2) == "0x")
    {
      the_value.erase(0, 2);
    }

  typet type;
  type = unsignedbv_typet(n);

  // e.g. 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984
  BigInt hex_addr = string2integer(the_value, 16);
  exprt the_val;
  the_val = constant_exprt(
    integer2binary(hex_addr, bv_width(type)), integer2string(hex_addr), type);

  dest.swap(the_val);
  set_sol_bytesn_size(dest.type(), the_value.length() / 2);
  return false;
}

// TODO: Float literal.
//    - Note: Currently Solidity does NOT support floating point data types or
//    fp arithmetic.
//      Everything is done in fixed-point arithmetic as of Solidity compiler
//      v0.8.6.
