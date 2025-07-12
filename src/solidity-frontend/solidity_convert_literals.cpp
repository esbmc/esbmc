#include <solidity-frontend/solidity_convert.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/ieee_float.h>
#include <util/string_constant.h>
#include <util/std_expr.h>

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

  if (the_value.find(".") != std::string::npos)
    return true;

  // Handle scientific notation, e.g., "1e2" -> "100"
  std::size_t e_pos = the_value.find_first_of("eE");
  if (e_pos != std::string::npos)
  {
    std::string base_part = the_value.substr(0, e_pos);
    std::string exp_part = the_value.substr(e_pos + 1);

    // Convert base and exponent to BigInt
    BigInt base = string2integer(base_part);
    BigInt exponent = string2integer(exp_part);

    // Calculate base * (10 ^ exponent)
    BigInt scale = ::power(BigInt(10), exponent);
    BigInt result = base * scale;

    the_value = integer2string(result);
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
          * value: 0000000000000000000000000000000000000000000000000000000000000101
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
  dest.type().set("#sol_type", "STRING_LITERAL");
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
  dest.type().set("#sol_bytesn_size", the_value.length()/2);
  return false;
}

// TODO: Float literal.
//    - Note: Currently Solidity does NOT support floating point data types or fp arithmetic.
//      Everything is done in fixed-point arithmetic as of Solidity compiler v0.8.6.
