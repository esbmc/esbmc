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

