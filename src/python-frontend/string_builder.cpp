#include "string_builder.h"
#include "python_converter.h"
#include "type_handler.h"
#include <util/arith_tools.h>
#include <util/std_code.h>
#include <util/expr_util.h>

string_builder::string_builder(python_converter &conv, string_handler *handler)
  : converter_(conv), str_handler_(handler)
{
}

contextt &string_builder::get_symbol_table() const
{
  return converter_.symbol_table();
}

type_handler &string_builder::get_type_handler()
{
  return converter_.get_type_handler();
}

exprt string_builder::make_null_terminator()
{
  BigInt zero(0);
  return constant_exprt(
    integer2binary(zero, 8), integer2string(zero), char_type());
}

exprt string_builder::make_char_constant(unsigned char ch)
{
  BigInt char_val(ch);
  return constant_exprt(
    integer2binary(char_val, 8), integer2string(char_val), char_type());
}

std::vector<exprt> string_builder::extract_string_chars(
  const exprt &expr,
  const nlohmann::json &json_node)
{
  std::vector<exprt> chars;

  // Handle JSON string constants first
  if (
    !json_node.is_null() && json_node.contains("_type") &&
    json_node["_type"] == "Constant" && json_node.contains("value"))
  {
    std::string str_value = json_node["value"].get<std::string>();
    chars.reserve(str_value.size());
    for (char c : str_value)
    {
      if (c == 0)
        break;
      chars.push_back(make_char_constant(static_cast<unsigned char>(c)));
    }
    return chars;
  }

  // Handle symbol expressions
  if (expr.is_symbol())
  {
    symbolt *symbol =
      get_symbol_table().find_symbol(expr.identifier().as_string());
    if (
      symbol && symbol->value.is_constant() && symbol->value.type().is_array())
    {
      for (size_t i = 0; i < symbol->value.operands().size(); ++i)
      {
        const exprt &ch = symbol->value.operands()[i];
        if (ch.is_constant())
        {
          try
          {
            BigInt char_val =
              binary2integer(ch.value().as_string(), ch.type().is_signedbv());
            if (char_val == 0)
              break;
            chars.push_back(ch);
          }
          catch (...)
          {
            chars.push_back(ch);
          }
        }
      }
    }
    return chars;
  }

  // Handle constant arrays
  if (expr.is_constant() && expr.type().is_array())
  {
    for (size_t i = 0; i < expr.operands().size(); ++i)
    {
      const exprt &ch = expr.operands()[i];
      if (ch.is_constant())
      {
        try
        {
          BigInt char_val =
            binary2integer(ch.value().as_string(), ch.type().is_signedbv());
          if (char_val == 0)
            break;
          chars.push_back(ch);
        }
        catch (...)
        {
          chars.push_back(ch);
        }
      }
    }
    return chars;
  }

  // Handle single character constants
  if (
    expr.is_constant() &&
    (expr.type().is_signedbv() || expr.type().is_unsignedbv()))
  {
    try
    {
      BigInt char_val =
        binary2integer(expr.value().as_string(), expr.type().is_signedbv());
      if (char_val != 0)
        chars.push_back(expr);
    }
    catch (...)
    {
      chars.push_back(expr);
    }
  }

  return chars;
}

exprt string_builder::build_null_terminated_string(
  const std::vector<exprt> &chars)
{
  // Calculate total size including null terminator
  size_t total_size = chars.size() + 1;

  // Create array type and expression
  typet string_type = get_type_handler().build_array(char_type(), total_size);
  exprt result = constant_exprt(
    array_typet(char_type(), from_integer(total_size, size_type())));
  result.type() = string_type;
  result.operands().resize(total_size);

  // Copy characters
  for (size_t i = 0; i < chars.size(); ++i)
    result.operands().at(i) = chars[i];

  // Add null terminator
  result.operands().at(chars.size()) = make_null_terminator();

  return result;
}

exprt string_builder::build_string_literal(const std::string &str)
{
  std::vector<exprt> chars;
  chars.reserve(str.size());
  for (char c : str)
    chars.push_back(make_char_constant(static_cast<unsigned char>(c)));

  return build_null_terminated_string(chars);
}

exprt string_builder::build_byte_string(const std::vector<uint8_t> &bytes)
{
  std::vector<exprt> chars;
  chars.reserve(bytes.size());
  for (uint8_t byte : bytes)
    chars.push_back(make_char_constant(byte));

  return build_null_terminated_string(chars);
}

exprt string_builder::ensure_null_terminated_string(exprt &e)
{
  // Already a proper string array - return as is
  if (e.type().is_array() && e.type().subtype() == char_type())
    return e;

  // Single character constant - convert to null-terminated string
  if (e.is_constant() && (e.type().is_signedbv() || e.type().is_unsignedbv()))
  {
    BigInt char_val =
      binary2integer(e.value().as_string(), e.type().is_signedbv());

    std::vector<exprt> chars;
    chars.push_back(
      make_char_constant(static_cast<unsigned char>(char_val.to_uint64())));
    return build_null_terminated_string(chars);
  }

  // For other types, fallback to converter's ensure_string_array
  str_handler_->ensure_string_array(e);
  return e;
}

exprt string_builder::concatenate_strings(
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right)
{
  // Handle edge cases with empty strings
  bool lhs_is_empty =
    str_handler_->is_zero_length_array(lhs) ||
    (lhs.is_constant() && lhs.type().is_array() && lhs.operands().size() <= 1);
  bool rhs_is_empty =
    str_handler_->is_zero_length_array(rhs) ||
    (rhs.is_constant() && rhs.type().is_array() && rhs.operands().size() <= 1);

  if (lhs_is_empty && rhs_is_empty)
    return lhs;
  if (lhs_is_empty && !rhs_is_empty)
    return rhs;
  if (!lhs_is_empty && rhs_is_empty)
    return lhs;

  // Extract characters from both operands
  std::vector<exprt> lhs_chars = extract_string_chars(lhs, left);
  std::vector<exprt> rhs_chars = extract_string_chars(rhs, right);

  // Combine character vectors
  std::vector<exprt> combined_chars;
  combined_chars.reserve(lhs_chars.size() + rhs_chars.size());
  combined_chars.insert(
    combined_chars.end(), lhs_chars.begin(), lhs_chars.end());
  combined_chars.insert(
    combined_chars.end(), rhs_chars.begin(), rhs_chars.end());

  // Build null-terminated result
  return build_null_terminated_string(combined_chars);
}

exprt string_builder::build_raw_byte_array(const std::vector<uint8_t> &bytes)
{
  // Get the proper bytes type from type_handler
  typet bytes_type = get_type_handler().get_typet("bytes", bytes.size());

  // Create zero-initialized array
  exprt result = gen_zero(bytes_type);

  // Get the element type from the bytes type's subtype
  const typet &element_type = bytes_type.subtype();

  // Copy bytes using the proper element type and width
  for (size_t i = 0; i < bytes.size(); ++i)
  {
    uint8_t byte = bytes[i];
    exprt byte_value = constant_exprt(
      integer2binary(BigInt(byte), bv_width(element_type)),
      integer2string(BigInt(byte)),
      element_type);
    result.operands().at(i) = byte_value;
  }

  return result;
}
