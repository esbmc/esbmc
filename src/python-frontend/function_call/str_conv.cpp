#include <python-frontend/function_call/expr.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/python_exception_handler.h>
#include <python-frontend/string_builder.h>
#include <python-frontend/string_handler.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/type_handler.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/ieee_float.h>
#include <util/message.h>
#include <util/python_types.h>
#include <util/std_expr.h>
#include <util/string_constant.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <sstream>
#include <stdexcept>

using namespace json_utils;

namespace
{
// Constants for UTF-8 encoding
constexpr unsigned int UTF8_1_BYTE_MAX = 0x7F;
constexpr unsigned int UTF8_2_BYTE_MAX = 0x7FF;
constexpr unsigned int UTF8_3_BYTE_MAX = 0xFFFF;
constexpr unsigned int UTF8_4_BYTE_MAX = 0x10FFFF;
constexpr unsigned int SURROGATE_START = 0xD800;
constexpr unsigned int SURROGATE_END = 0xDFFF;

static std::string utf8_encode(unsigned int int_value)
{
  /**
   * Convert an integer value into its UTF-8 character equivalent
   * similar to the python chr() function
   */

  // Check for surrogate pairs (invalid in UTF-8)
  if (int_value >= SURROGATE_START && int_value <= SURROGATE_END)
  {
    std::ostringstream oss;
    oss << "Code point 0x" << std::hex << std::uppercase << int_value
        << " is a surrogate pair, invalid in UTF-8";
    throw std::invalid_argument(oss.str());
  }

  // Manual UTF-8 encoding
  std::string char_out;

  if (int_value <= UTF8_1_BYTE_MAX)
    char_out.append(1, static_cast<char>(int_value));
  else if (int_value <= UTF8_2_BYTE_MAX)
  {
    char_out.append(1, static_cast<char>(0xc0 | ((int_value >> 6) & 0x1f)));
    char_out.append(1, static_cast<char>(0x80 | (int_value & 0x3f)));
  }
  else if (int_value <= UTF8_3_BYTE_MAX)
  {
    char_out.append(1, static_cast<char>(0xe0 | ((int_value >> 12) & 0x0f)));
    char_out.append(1, static_cast<char>(0x80 | ((int_value >> 6) & 0x3f)));
    char_out.append(1, static_cast<char>(0x80 | (int_value & 0x3f)));
  }
  else if (int_value <= UTF8_4_BYTE_MAX)
  {
    char_out.append(1, static_cast<char>(0xf0 | ((int_value >> 18) & 0x07)));
    char_out.append(1, static_cast<char>(0x80 | ((int_value >> 12) & 0x3f)));
    char_out.append(1, static_cast<char>(0x80 | ((int_value >> 6) & 0x3f)));
    char_out.append(1, static_cast<char>(0x80 | (int_value & 0x3f)));
  }
  else
  {
    std::ostringstream oss;
    oss << "argument '0x" << std::hex << std::uppercase << int_value
        << "' outside of Unicode range: [0x000000, 0x110000)";
    throw std::out_of_range(oss.str());
  }
  return char_out;
}
} // namespace

int function_call_expr::decode_utf8_codepoint(const std::string &utf8_str) const
{
  if (utf8_str.empty())
  {
    throw std::runtime_error(
      "TypeError: expected a character, but string of length 0 found");
  }

  const unsigned char *bytes =
    reinterpret_cast<const unsigned char *>(utf8_str.c_str());
  size_t length = utf8_str.length();

  // Manual UTF-8 decoding
  if ((bytes[0] & 0x80) == 0)
  {
    // 1-byte ASCII
    if (length != 1)
      throw std::runtime_error("TypeError: expected a single character");
    return bytes[0];
  }
  else if ((bytes[0] & 0xE0) == 0xC0)
  {
    // 2-byte sequence
    if (length != 2)
      throw std::runtime_error("TypeError: expected a single character");
    return ((bytes[0] & 0x1F) << 6) | (bytes[1] & 0x3F);
  }
  else if ((bytes[0] & 0xF0) == 0xE0)
  {
    // 3-byte sequence
    if (length != 3)
      throw std::runtime_error("TypeError: expected a single character");
    return ((bytes[0] & 0x0F) << 12) | ((bytes[1] & 0x3F) << 6) |
           (bytes[2] & 0x3F);
  }
  else if ((bytes[0] & 0xF8) == 0xF0)
  {
    // 4-byte sequence
    if (length != 4)
      throw std::runtime_error("TypeError: expected a single character");
    return ((bytes[0] & 0x07) << 18) | ((bytes[1] & 0x3F) << 12) |
           ((bytes[2] & 0x3F) << 6) | (bytes[3] & 0x3F);
  }
  else
    throw std::runtime_error("ValueError: received invalid UTF-8 input");
}

exprt function_call_expr::handle_chr(nlohmann::json &arg) const
{
  int int_value = 0;
  bool is_constant = false;

  // Check for unary minus: e.g., chr(-1)
  if (arg.contains("_type") && arg["_type"] == "UnaryOp")
  {
    const auto &op = arg["op"];
    const auto &operand = arg["operand"];

    if (
      op["_type"] == "USub" && operand.contains("value") &&
      operand["value"].is_number_integer())
    {
      int_value = -operand["value"].get<int>();
      is_constant = true;
    }
    else
      return converter_.get_exception_handler().gen_exception_raise(
        "TypeError", "Unsupported UnaryOp in chr()");
  }

  // Handle integer input
  else if (arg.contains("value") && arg["value"].is_number_integer())
  {
    int_value = arg["value"].get<int>();
    is_constant = true;
  }

  // Reject float input
  else if (arg.contains("value") && arg["value"].is_number_float())
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError", "chr() argument must be int, not float");

  // Try converting string input to integer
  else if (arg.contains("value") && arg["value"].is_string())
  {
    const std::string &s = arg["value"].get<std::string>();
    try
    {
      int_value = std::stoi(s);
      is_constant = true;
    }
    catch (const std::invalid_argument &)
    {
      return converter_.get_exception_handler().gen_exception_raise(
        "TypeError", "invalid string passed to chr()");
    }
  }

  // Handle variable references (Name nodes)
  else if (arg.contains("_type") && arg["_type"] == "Name")
  {
    const symbolt *sym = lookup_python_symbol(arg["id"]);
    if (!sym)
    {
      // Symbol not found - use runtime conversion via string_handler
      exprt var_expr = converter_.get_expr(arg);
      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_chr_conversion(
        var_expr, loc);
    }

    exprt val = sym->value;

    if (!val.is_constant())
      val = converter_.get_resolved_value(val);

    if (val.is_nil())
    {
      // Runtime variable: use string_handler for runtime conversion
      exprt var_expr = converter_.get_expr(arg);
      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_chr_conversion(
        var_expr, loc);
    }

    // Even if the symbol has a constant value, we should not
    // perform compile-time conversion if the symbol is mutable (lvalue)
    // because its value may change at runtime (e.g., loop variables)
    if (sym->lvalue)
    {
      // This is a mutable variable - use runtime conversion
      exprt var_expr = converter_.get_expr(arg);
      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_chr_conversion(
        var_expr, loc);
    }

    // Try to extract constant value (only for truly constant symbols)
    const auto &const_expr = to_constant_expr(val);
    std::string binary_str = id2string(const_expr.get_value());
    try
    {
      int_value = std::stoul(binary_str, nullptr, 2);
      is_constant = true;
    }
    catch (std::out_of_range &)
    {
      return converter_.get_exception_handler().gen_exception_raise(
        "ValueError", "chr() argument outside of Unicode range");
    }
    catch (std::invalid_argument &)
    {
      return converter_.get_exception_handler().gen_exception_raise(
        "TypeError", "must be of type int");
    }

    arg["_type"] = "Constant";
    arg.erase("id");
    arg.erase("ctx");
  }

  // If we have a constant value, do compile-time conversion
  if (is_constant)
  {
    std::string utf8_encoded;

    try
    {
      utf8_encoded = utf8_encode(int_value);
    }
    catch (const std::out_of_range &e)
    {
      return converter_.get_exception_handler().gen_exception_raise(
        "ValueError", "chr()");
    }

    // Build a proper character array, not a single char
    // Create array type with proper size (length + null terminator)
    size_t array_size = utf8_encoded.size() + 1;
    typet char_array_type =
      array_typet(char_type(), from_integer(array_size, size_type()));

    // Build the array expression with all characters
    std::vector<unsigned char> char_data(
      utf8_encoded.begin(), utf8_encoded.end());
    char_data.push_back('\0'); // Add null terminator

    // Use converter's make_char_array_expr to build proper array
    exprt result = converter_.make_char_array_expr(char_data, char_array_type);

    return result;
  }

  // If we reach here, we need runtime conversion
  exprt var_expr = converter_.get_expr(arg);
  locationt loc = converter_.get_location_from_decl(call_);
  return converter_.get_string_handler().handle_chr_conversion(var_expr, loc);
}

exprt function_call_expr::handle_ord(nlohmann::json &arg) const
{
  int code_point = 0;

  // Ensure the argument is a string
  if (is_string_arg(arg))
  {
    const std::string &s = arg["value"].get<std::string>();
    code_point = decode_utf8_codepoint(s);
  }
  // Handle ord with symbol
  else if (arg["_type"] == "Name" && arg.contains("id"))
  {
    const symbolt *sym = lookup_python_symbol(arg["id"]);
    if (!sym)
    {
      std::string var_name = arg["id"].get<std::string>();
      return converter_.get_exception_handler().gen_exception_raise(
        "NameError", "variable '" + var_name + "' is not defined");
    }

    typet operand_type = sym->value.type();
    std::string py_type = type_handler_.type_to_string(operand_type);

    if (operand_type != char_type() && py_type != "str")
    {
      return converter_.get_exception_handler().gen_exception_raise(
        "TypeError",
        "ord() expected string of length 1, but " + py_type + " found");
    }

    // For runtime variables (mutable), try to extract constant value if available
    if (sym->lvalue && !sym->value.is_nil())
    {
      auto value_opt = extract_string_from_symbol(sym);
      if (value_opt)
      {
        // Successfully extracted constant value from lvalue string
        code_point = decode_utf8_codepoint(*value_opt);

        // Remove Name data
        arg["_type"] = "Constant";
        arg.erase("id");
        arg.erase("ctx");

        // Replace the arg with the integer value
        arg["value"] = code_point;
        arg["type"] = "int";

        // Build and return the integer expression
        exprt expr = converter_.get_expr(arg);
        expr.type() = type_handler_.get_typet("int", 0);
        return expr;
      }
      // If extraction failed for lvalue, fall through to runtime conversion
    }

    // Use runtime conversion for variables without constant value or failed extraction
    if (sym->value.is_nil() || sym->lvalue)
    {
      exprt var_expr = converter_.get_expr(arg);

      if (
        var_expr.type() == char_type() || var_expr.type().is_signedbv() ||
        var_expr.type().is_unsignedbv())
      {
        return typecast_exprt(var_expr, int_type());
      }

      return converter_.get_exception_handler().gen_exception_raise(
        "ValueError", "ord() requires a character");
    }

    // Compile-time extraction for constant symbols
    auto value_opt = extract_string_from_symbol(sym);
    if (!value_opt)
    {
      return converter_.get_exception_handler().gen_exception_raise(
        "ValueError", "failed to extract string from symbol");
    }

    code_point = decode_utf8_codepoint(*value_opt);

    // Remove Name data
    arg["_type"] = "Constant";
    arg.erase("id");
    arg.erase("ctx");
  }
  else
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError", "ord() argument must be a string");

  // Replace the arg with the integer value
  arg["value"] = code_point;
  arg["type"] = "int";

  // Build and return the integer expression
  exprt expr = converter_.get_expr(arg);
  expr.type() = type_handler_.get_typet("int", 0);
  return expr;
}


exprt function_call_expr::handle_int_to_str(nlohmann::json &arg) const
{
  std::string str_val = std::to_string(arg["value"].get<int>());
  typet t = type_handler_.get_typet("str", str_val.size() + 1);
  return converter_.make_char_array_expr(
    std::vector<uint8_t>(str_val.begin(), str_val.end()), t);
}

exprt function_call_expr::handle_int_to_bytes() const
{
  const auto &args = call_["args"];
  // Python accepts both int.to_bytes(x, ...) and x.to_bytes(...).
  const bool is_type_method_call = call_["func"]["_type"] == "Attribute" &&
                                   call_["func"]["value"]["_type"] == "Name" &&
                                   call_["func"]["value"]["id"] == "int";

  // In the type-method form, the integer value is passed explicitly as the
  // first argument. In the instance-method form, it comes from the receiver.
  if (
    args.size() < (is_type_method_call ? 3 : 2) ||
    args.size() > (is_type_method_call ? 4 : 3))
  {
    throw std::runtime_error(
      is_type_method_call ? "int.to_bytes() expects 3 or 4 positional "
                            "arguments"
                          : "int.to_bytes() expects 2 or 3 positional "
                            "arguments");
  }

  exprt value = is_type_method_call
                  ? converter_.get_expr(args[0])
                  : converter_.get_expr(call_["func"]["value"]);
  const nlohmann::json &length_arg = args[is_type_method_call ? 1 : 0];
  const nlohmann::json &byteorder_arg = args[is_type_method_call ? 2 : 1];

  if (
    !length_arg.contains("value") || !length_arg["value"].is_number_unsigned())
    throw std::runtime_error(
      "int.to_bytes() currently expects a constant unsigned length");

  const std::size_t length = length_arg["value"].get<std::size_t>();

  bool big_endian = true;
  if (byteorder_arg.contains("value"))
  {
    if (byteorder_arg["value"].is_boolean())
      big_endian = byteorder_arg["value"].get<bool>();
    else if (byteorder_arg["value"].is_string())
      big_endian = byteorder_arg["value"].get<std::string>() == "big";
  }

  const typet bytes_type = type_handler_.get_typet("bytes", length);
  exprt result = gen_zero(bytes_type);
  const typet &elem_type = bytes_type.subtype();

  if (!value.type().is_unsignedbv())
  {
    // Convert the source value to an unsigned integer type before extracting
    // individual bytes with shifts and masks.
    const unsigned width =
      (value.type().is_signedbv() || value.type().is_unsignedbv())
        ? std::max(1u, bv_width(value.type()))
        : 64;
    value = typecast_exprt(value, unsignedbv_typet(width));
  }

  // Fill the output array one byte at a time. For big-endian we start from the
  // most significant byte; for little-endian we start from the least significant one.
  for (std::size_t i = 0; i < length; ++i)
  {
    const std::size_t byte_index = big_endian ? (length - 1 - i) : i;

    // Shift the selected byte down to the low 8 bits and mask everything else out.
    const exprt shift_amount = from_integer(byte_index * 8, value.type());
    exprt shifted("shr", value.type());
    shifted.copy_to_operands(value, shift_amount);

    exprt masked("bitand", value.type());
    masked.copy_to_operands(shifted, from_integer(0xff, value.type()));

    result.operands().at(i) = typecast_exprt(masked, elem_type);
  }

  return result;
}

exprt function_call_expr::handle_float_to_str(nlohmann::json &arg) const
{
  std::string str_val = std::to_string(arg["value"].get<double>());

  // Remove unnecessary trailing zeros and dot if needed (to match Python str behavior)
  // Example: "5.500000" → "5.5"
  str_val.erase(str_val.find_last_not_of('0') + 1, std::string::npos);
  if (str_val.back() == '.')
    str_val.pop_back();

  typet t = type_handler_.get_typet("str", str_val.size() + 1);
  return converter_.make_char_array_expr(
    std::vector<uint8_t>(str_val.begin(), str_val.end()), t);
}

exprt function_call_expr::handle_complex_to_str() const
{
  // Non-constant complex: fall back to a generic placeholder.
  return converter_.get_string_builder().build_string_literal("(complex)");
}

void function_call_expr::handle_float_to_int(nlohmann::json &arg) const
{
  double value = arg["value"].get<double>();
  arg["value"] = static_cast<int>(value);
}

void function_call_expr::handle_int_to_float(nlohmann::json &arg) const
{
  int value = arg["value"].get<int>();
  arg["value"] = static_cast<double>(value);
}
