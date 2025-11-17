#include <python-frontend/string_handler.h>
#include <python-frontend/char_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/string_builder.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/symbol_id.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/python_types.h>
#include <util/std_expr.h>
#include <util/std_code.h>
#include <util/string_constant.h>
#include <util/symbol.h>
#include <util/type.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>

string_handler::string_handler(
  python_converter &converter,
  contextt &symbol_table,
  type_handler &type_handler,
  string_builder *str_builder)
  : converter_(converter),
    symbol_table_(symbol_table),
    type_handler_(type_handler),
    string_builder_(str_builder)
{
}

BigInt string_handler::get_string_size(const exprt &expr)
{
  if (!expr.type().is_array())
  {
    // For non-array types in f-strings, convert them first to get actual size
    if (expr.is_constant() && type_utils::is_integer_type(expr.type()))
    {
      // Convert the actual integer to string to get real size
      BigInt value =
        binary2integer(expr.value().as_string(), expr.type().is_signedbv());
      std::string str_repr = std::to_string(value.to_int64());
      return BigInt(str_repr.size() + 1); // +1 for null terminator
    }

    if (expr.is_symbol())
    {
      const symbolt *symbol = symbol_table_.find_symbol(expr.identifier());
      if (symbol && symbol->type.is_array())
      {
        const auto &arr_type = to_array_type(symbol->type);
        return binary2integer(arr_type.size().value().as_string(), false);
      }
      // For non-array symbols, we need a reasonable default since we can't compute actual size
      return BigInt(20); // Conservative default
    }

    // For other types, use conservative defaults
    if (expr.type().is_bool())
      return BigInt(6); // "False" + null terminator

    // Default fallback
    return BigInt(20);
  }

  const auto &arr_type = to_array_type(expr.type());
  return binary2integer(arr_type.size().value().as_string(), false);
}

std::string
string_handler::process_format_spec(const nlohmann::json &format_spec)
{
  if (format_spec.is_null() || !format_spec.contains("_type"))
    return "";

  // Handle direct Constant format spec
  if (format_spec["_type"] == "Constant" && format_spec.contains("value"))
    return format_spec["value"].get<std::string>();

  // Handle JoinedStr format spec (which contains Constant values)
  if (format_spec["_type"] == "JoinedStr" && format_spec.contains("values"))
  {
    std::string result;
    for (const auto &value : format_spec["values"])
      if (value["_type"] == "Constant" && value.contains("value"))
        result += value["value"].get<std::string>();
    return result;
  }

  // Log warning for unsupported format specifications
  std::string spec_type = format_spec.contains("_type")
                            ? format_spec["_type"].get<std::string>()
                            : "unknown";
  log_warning("Unsupported f-string format specification type: {}", spec_type);

  return "";
}

std::string string_handler::float_to_string(
  const std::string &float_bits,
  std::size_t width,
  int precision)
{
  double val = 0.0;

  if (width == 32 && float_bits.length() == 32)
  {
    // IEEE 754 single precision
    uint32_t bits = 0;
    for (std::size_t i = 0; i < width; ++i)
      if (float_bits[i] == '1')
        bits |= (1U << (width - 1 - i));

    float float_val;
    std::memcpy(&float_val, &bits, sizeof(float));
    val = static_cast<double>(float_val);
  }
  else if (width == 64 && float_bits.length() == 64)
  {
    // IEEE 754 double precision
    uint64_t bits = 0;
    for (std::size_t i = 0; i < width; ++i)
      if (float_bits[i] == '1')
        bits |= (1ULL << (width - 1 - i));

    std::memcpy(&val, &bits, sizeof(double));
  }
  else
  {
    throw std::runtime_error("Invalid float bit width");
  }

  // Use proper rounding to avoid IEEE 754 precision issues
  double multiplier = std::pow(10.0, precision);
  double rounded = std::round(val * multiplier) / multiplier;

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << rounded;
  return oss.str();
}

exprt string_handler::apply_format_specification(
  const exprt &expr,
  const std::string &format)
{
  // Basic format specification handling
  if (format.empty())
    return convert_to_string(expr);

  // Handle integer formatting
  if (format == "d" || format == "i")
    return convert_to_string(expr);

  // Handle float formatting with precision
  else if (format.find(".") != std::string::npos && format.back() == 'f')
  {
    // Extract precision from format string (e.g., ".2f" -> 2)
    size_t dot_pos = format.find(".");
    size_t f_pos = format.find("f");
    if (
      dot_pos != std::string::npos && f_pos != std::string::npos &&
      f_pos > dot_pos)
    {
      std::string precision_str =
        format.substr(dot_pos + 1, f_pos - dot_pos - 1);
      int precision = 6; // default
      try
      {
        precision = std::stoi(precision_str);
      }
      catch (...)
      {
        precision = 6;
      }

      // Handle floatbv expressions (both constant and symbols)
      if (expr.type().is_floatbv())
      {
        const typet &t = expr.type();
        const std::size_t float_width = bv_width(t);

        // Support common floating point widths
        if (t.is_floatbv() && (float_width == 32 || float_width == 64))
        {
          const std::string *float_bits = nullptr;

          // Handle constant expressions
          if (expr.is_constant())
            float_bits = &expr.value().as_string();
          // Handle symbol expressions
          else if (expr.is_symbol())
          {
            const symbol_exprt &sym_expr = to_symbol_expr(expr);
            const symbolt *symbol =
              symbol_table_.find_symbol(sym_expr.get_identifier());

            if (symbol && symbol->value.is_constant())
              float_bits = &symbol->value.value().as_string();
          }

          if (float_bits && float_bits->length() == float_width)
          {
            std::string formatted_str =
              float_to_string(*float_bits, float_width, precision);

            typet string_type =
              type_handler_.build_array(char_type(), formatted_str.size() + 1);
            std::vector<unsigned char> chars(
              formatted_str.begin(), formatted_str.end());
            chars.push_back('\0');

            return make_char_array_expr(chars, string_type);
          }
        }
      }
    }
  }

  // Default: just convert to string
  return convert_to_string(expr);
}

exprt string_handler::make_char_array_expr(
  const std::vector<unsigned char> &chars,
  const typet &type)
{
  exprt arr = gen_zero(type);
  for (size_t i = 0; i < chars.size() && i < arr.operands().size(); ++i)
  {
    arr.operands()[i] = from_integer(chars[i], char_type());
  }
  return arr;
}

exprt string_handler::convert_to_string(const exprt &expr)
{
  const typet &t = expr.type();

  // Already a string/char array - return as is
  if (t.is_array() && t.subtype() == char_type())
    return expr;

  // Handle symbol references
  if (expr.is_symbol())
  {
    const symbolt *symbol = symbol_table_.find_symbol(expr.identifier());
    if (symbol)
    {
      // If symbol has string type, return it
      if (symbol->type.is_array() && symbol->type.subtype() == char_type())
        return expr;

      // If symbol has a constant value, convert that
      if (symbol->value.is_constant())
        return convert_to_string(symbol->value);
    }
  }

  // Handle constants
  if (expr.is_constant())
  {
    if (type_utils::is_integer_type(t))
    {
      BigInt value = binary2integer(expr.value().as_string(), t.is_signedbv());
      std::string str_value = std::to_string(value.to_int64());

      typet string_type =
        type_handler_.build_array(char_type(), str_value.size() + 1);
      std::vector<unsigned char> chars(str_value.begin(), str_value.end());
      chars.push_back('\0'); // null terminator

      return make_char_array_expr(chars, string_type);
    }
    else if (t.is_floatbv())
    {
      std::string str_value = "0.0";
      if (expr.is_constant() && !expr.value().empty())
      {
        const std::string &float_bits = expr.value().as_string();
        if (t.is_floatbv() && bv_width(t) == 64 && float_bits.length() == 64)
        {
          str_value = float_to_string(float_bits, 64, 6);
        }
      }

      typet string_type =
        type_handler_.build_array(char_type(), str_value.size() + 1);
      std::vector<unsigned char> chars(str_value.begin(), str_value.end());
      chars.push_back('\0');

      return make_char_array_expr(chars, string_type);
    }
    else if (t.is_bool())
    {
      // Convert boolean to string
      bool value = expr.is_true();
      std::string str_value = value ? "True" : "False";

      typet string_type =
        type_handler_.build_array(char_type(), str_value.size() + 1);
      std::vector<unsigned char> chars(str_value.begin(), str_value.end());
      chars.push_back('\0');

      return make_char_array_expr(chars, string_type);
    }
  }

  // For non-constant expressions, we'd need runtime conversion
  // For now, create a placeholder string
  std::string placeholder = "<expr>";
  typet string_type =
    type_handler_.build_array(char_type(), placeholder.size() + 1);
  std::vector<unsigned char> chars(placeholder.begin(), placeholder.end());
  chars.push_back('\0');

  return make_char_array_expr(chars, string_type);
}

exprt string_handler::get_fstring_expr(const nlohmann::json &element)
{
  if (!element.contains("values") || element["values"].empty())
  {
    // Empty f-string
    typet empty_string_type = type_handler_.build_array(char_type(), 1);
    exprt empty_str = gen_zero(empty_string_type);
    empty_str.operands().at(0) = from_integer(0, char_type());
    return empty_str;
  }

  const auto &values = element["values"];
  std::vector<exprt> parts;
  BigInt total_estimated_size = BigInt(1); // Start with 1 for null terminator

  for (const auto &value : values)
  {
    exprt part_expr;

    try
    {
      if (value["_type"] == "Constant")
      {
        // String literal part - delegate to converter
        part_expr = converter_.get_literal(value);
      }
      else if (value["_type"] == "FormattedValue")
      {
        // Expression to be formatted
        exprt expr = converter_.get_expr(value["value"]);

        // Handle format specification if present
        if (value.contains("format_spec") && !value["format_spec"].is_null())
        {
          std::string format = process_format_spec(value["format_spec"]);
          part_expr = apply_format_specification(expr, format);
        }
        else
          part_expr = convert_to_string(expr);
      }
      else
      {
        // Other expression types
        exprt expr = converter_.get_expr(value);
        part_expr = convert_to_string(expr);
      }

      parts.push_back(part_expr);
      total_estimated_size += get_string_size(part_expr) -
                              1; // -1 to avoid double counting terminators
    }
    catch (const std::exception &e)
    {
      log_warning("Error processing f-string part: {}", e.what());
      // Create error placeholder
      std::string error_str = "<e>";
      typet error_type =
        type_handler_.build_array(char_type(), error_str.size() + 1);
      std::vector<unsigned char> chars(error_str.begin(), error_str.end());
      chars.push_back('\0');
      parts.push_back(make_char_array_expr(chars, error_type));
      total_estimated_size += BigInt(error_str.size());
    }
  }

  // If only one part, return it directly
  if (parts.size() == 1)
    return parts[0];

  // Concatenate all parts
  exprt result = parts[0];
  for (size_t i = 1; i < parts.size(); ++i)
  {
    nlohmann::json empty_left, empty_right;
    result =
      handle_string_concatenation(result, parts[i], empty_left, empty_right);
  }

  return result;
}

exprt string_handler::handle_string_concatenation(
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right)
{
  return string_builder_->concatenate_strings(lhs, rhs, left, right);
}

exprt string_handler::handle_string_repetition(exprt &lhs, exprt &rhs)
{
  return string_builder_->handle_string_repetition(lhs, rhs);
}

bool string_handler::is_zero_length_array(const exprt &expr)
{
  if (expr.id() == "sideeffect")
    return false;

  if (!expr.type().is_array())
    return false;

  const auto &arr_type = to_array_type(expr.type());
  if (!arr_type.size().is_constant())
    return false;

  BigInt size = binary2integer(arr_type.size().value().as_string(), false);
  return size == 0;
}

std::string string_handler::extract_string_from_array_operands(
  const exprt &array_expr) const
{
  std::string result;
  for (const auto &op : array_expr.operands())
  {
    if (op.is_constant())
    {
      BigInt val =
        binary2integer(op.value().as_string(), op.type().is_signedbv());
      if (val == 0)
        break;
      result += static_cast<char>(val.to_uint64());
    }
  }
  return result;
}

void string_handler::ensure_string_array(exprt &expr)
{
  if (expr.type().is_pointer())
    return;

  if (!expr.type().is_array())
  {
    // Explicitly build the array and
    // ensure null-termination (size 2: char + \0)
    typet t = type_handler_.build_array(expr.type(), 2);
    exprt arr = gen_zero(t);
    arr.operands().at(0) = expr;
    expr = arr;
  }
}

exprt string_handler::handle_string_operations(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right,
  const nlohmann::json &element)
{
  if (op == "Mult")
    return handle_string_repetition(lhs, rhs);

  ensure_string_array(lhs);
  ensure_string_array(rhs);

  assert(lhs.type().is_array() || lhs.type().is_pointer());
  assert(rhs.type().is_array() || rhs.type().is_pointer());

  if (op == "Eq" || op == "NotEq")
    return handle_string_comparison(op, lhs, rhs, element);
  else if (op == "Add")
    return handle_string_concatenation(lhs, rhs, left, right);

  return nil_exprt();
}

exprt string_handler::get_array_base_address(const exprt &arr)
{
  exprt index = index_exprt(arr, from_integer(0, index_type()));
  return address_of_exprt(index);
}

exprt string_handler::handle_string_concatenation_with_promotion(
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right)
{
  if (lhs.type().is_array() && !rhs.type().is_array())
  {
    // LHS is array, RHS is single char - promote RHS to string array
    if (type_utils::is_integer_type(rhs.type()))
    {
      typet string_type = type_handler_.build_array(char_type(), 2);
      exprt str_array = gen_zero(string_type);
      str_array.operands().at(0) = rhs;
      str_array.operands().at(1) = gen_zero(char_type()); // null terminator
      rhs = str_array;
    }
  }
  else if (!lhs.type().is_array() && rhs.type().is_array())
  {
    // RHS is array, LHS is single char - promote LHS to string array
    if (type_utils::is_integer_type(lhs.type()))
    {
      typet string_type = type_handler_.build_array(char_type(), 2);
      exprt str_array = gen_zero(string_type);
      str_array.operands().at(0) = lhs;
      str_array.operands().at(1) = gen_zero(char_type()); // null terminator
      lhs = str_array;
    }
  }

  return handle_string_concatenation(lhs, rhs, left, right);
}

exprt string_handler::ensure_null_terminated_string(exprt &e)
{
  return string_builder_->ensure_null_terminated_string(e);
}

exprt string_handler::handle_string_comparison(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &element)
{
  return converter_.handle_string_comparison(op, lhs, rhs, element);
}

std::string string_handler::ensure_string_function_symbol(
  const std::string &function_name,
  const typet &return_type,
  const std::vector<typet> &arg_types,
  const locationt &location)
{
  symbol_id func_id;
  func_id.set_prefix("c:");
  func_id.set_function(function_name);

  std::string func_symbol_id = func_id.to_string();

  if (symbol_table_.find_symbol(func_symbol_id.c_str()) == nullptr)
  {
    code_typet code_type;
    code_type.return_type() = return_type;

    for (const auto &arg_type : arg_types)
    {
      code_typet::argumentt arg;
      arg.type() = arg_type;
      code_type.arguments().push_back(arg);
    }

    symbolt symbol = converter_.create_symbol(
      "", function_name, func_symbol_id, location, code_type);

    converter_.add_symbol(symbol);
  }

  return func_symbol_id;
}

exprt string_handler::handle_string_startswith(
  const exprt &string_obj,
  const exprt &prefix_arg,
  const locationt &location)
{
  // Ensure both are proper null-terminated strings
  exprt string_copy = string_obj;
  exprt prefix_copy = prefix_arg;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt prefix_expr = ensure_null_terminated_string(prefix_copy);

  // Get string addresses
  exprt str_addr = get_array_base_address(str_expr);
  exprt prefix_addr = get_array_base_address(prefix_expr);

  // Calculate prefix length: len(prefix_expr) - 1 (exclude null terminator)
  const array_typet &prefix_type = to_array_type(prefix_expr.type());
  exprt prefix_len = prefix_type.size();

  // Subtract 1 for null terminator
  exprt one = from_integer(1, prefix_len.type());
  exprt actual_len("-", prefix_len.type());
  actual_len.copy_to_operands(prefix_len, one);

  // Find strncmp symbol
  symbolt *strncmp_symbol = symbol_table_.find_symbol("c:@F@strncmp");
  if (!strncmp_symbol)
    throw std::runtime_error("strncmp function not found for startswith()");

  // Call strncmp(str, prefix, len(prefix))
  side_effect_expr_function_callt strncmp_call;
  strncmp_call.function() = symbol_expr(*strncmp_symbol);
  strncmp_call.arguments() = {str_addr, prefix_addr, actual_len};
  strncmp_call.location() = location;
  strncmp_call.type() = int_type();

  // Check if result == 0 (strings match)
  exprt zero = gen_zero(int_type());
  exprt equal("=", bool_type());
  equal.copy_to_operands(strncmp_call, zero);

  return equal;
}

exprt string_handler::handle_string_endswith(
  const exprt &string_obj,
  const exprt &suffix_arg,
  const locationt &location)
{
  // Ensure both are proper null-terminated strings
  exprt string_copy = string_obj;
  exprt suffix_copy = suffix_arg;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt suffix_expr = ensure_null_terminated_string(suffix_copy);

  // Get string addresses
  // Handle both pointer and array types
  exprt str_addr;
  exprt suffix_addr;

  if (str_expr.type().is_pointer())
    str_addr = str_expr;
  else
    str_addr = get_array_base_address(str_expr);

  if (suffix_expr.type().is_pointer())
    suffix_addr = suffix_expr;
  else
    suffix_addr = get_array_base_address(suffix_expr);

  // For length calculation, we need to use strlen for pointer types
  // Find strlen symbol
  symbolt *strlen_symbol = symbol_table_.find_symbol("c:@F@strlen");
  if (!strlen_symbol)
    throw std::runtime_error("strlen function not found for endswith()");

  // Get string length using strlen
  side_effect_expr_function_callt str_strlen_call;
  str_strlen_call.function() = symbol_expr(*strlen_symbol);
  str_strlen_call.arguments() = {str_addr};
  str_strlen_call.location() = location;
  str_strlen_call.type() = size_type();

  // Get suffix length using strlen
  side_effect_expr_function_callt suffix_strlen_call;
  suffix_strlen_call.function() = symbol_expr(*strlen_symbol);
  suffix_strlen_call.arguments() = {suffix_addr};
  suffix_strlen_call.location() = location;
  suffix_strlen_call.type() = size_type();

  // Check if suffix is longer than string
  exprt len_check(">", bool_type());
  len_check.copy_to_operands(suffix_strlen_call, str_strlen_call);

  // Calculate offset: strlen(str) - strlen(suffix)
  exprt offset("-", size_type());
  offset.copy_to_operands(str_strlen_call, suffix_strlen_call);

  // Get pointer to the position: str + offset
  exprt offset_ptr("+", gen_pointer_type(char_type()));
  offset_ptr.copy_to_operands(str_addr, offset);

  // Find strncmp symbol
  symbolt *strncmp_symbol = symbol_table_.find_symbol("c:@F@strncmp");
  if (!strncmp_symbol)
    throw std::runtime_error("strncmp function not found for endswith()");

  // Call strncmp(str + offset, suffix, strlen(suffix))
  side_effect_expr_function_callt strncmp_call;
  strncmp_call.function() = symbol_expr(*strncmp_symbol);
  strncmp_call.arguments() = {offset_ptr, suffix_addr, suffix_strlen_call};
  strncmp_call.location() = location;
  strncmp_call.type() = int_type();

  // Check if result == 0 (strings match)
  exprt zero = gen_zero(int_type());
  exprt strings_equal("=", bool_type());
  strings_equal.copy_to_operands(strncmp_call, zero);

  // Return: (suffix_len <= str_len) && (strncmp(...) == 0)
  exprt len_ok("not", bool_type());
  len_ok.copy_to_operands(len_check);

  exprt result("and", bool_type());
  result.copy_to_operands(len_ok, strings_equal);

  return result;
}

exprt string_handler::handle_string_isdigit(
  const exprt &string_obj,
  const locationt &location)
{
  // Check if this is a single character
  if (string_obj.type().is_unsignedbv() || string_obj.type().is_signedbv())
  {
    // Call Python's single-character version
    symbolt *isdigit_symbol =
      symbol_table_.find_symbol("c:@F@__python_char_isdigit");
    if (!isdigit_symbol)
      throw std::runtime_error(
        "__python_char_isdigit function not found in symbol table");

    side_effect_expr_function_callt isdigit_call;
    isdigit_call.function() = symbol_expr(*isdigit_symbol);
    isdigit_call.arguments().push_back(string_obj);
    isdigit_call.location() = location;
    isdigit_call.type() = bool_type();

    return isdigit_call;
  }

  // For full strings, use the string version
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);

  // Get base address of the string
  exprt str_addr = get_array_base_address(str_expr);

  // Find the helper function symbol
  symbolt *isdigit_str_symbol =
    symbol_table_.find_symbol("c:@F@__python_str_isdigit");
  if (!isdigit_str_symbol)
    throw std::runtime_error("str_isdigit function not found in symbol table");

  // Call str_isdigit(str) - returns bool (0 or 1)
  side_effect_expr_function_callt isdigit_call;
  isdigit_call.function() = symbol_expr(*isdigit_str_symbol);
  isdigit_call.arguments().push_back(str_addr);
  isdigit_call.location() = location;
  isdigit_call.type() = bool_type();

  return isdigit_call;
}

exprt string_handler::handle_string_isalpha(
  const exprt &string_obj,
  const locationt &location)
{
  // Check if this is a single character
  if (string_obj.type().is_unsignedbv() || string_obj.type().is_signedbv())
  {
    // Call Python's single-character version (not C's isalpha)
    symbolt *isalpha_symbol =
      symbol_table_.find_symbol("c:@F@__python_char_isalpha");
    if (!isalpha_symbol)
      throw std::runtime_error(
        "__python_char_isalpha function not found in symbol table");

    side_effect_expr_function_callt isalpha_call;
    isalpha_call.function() = symbol_expr(*isalpha_symbol);
    isalpha_call.arguments().push_back(string_obj);
    isalpha_call.location() = location;
    isalpha_call.type() = bool_type();

    return isalpha_call;
  }

  // For full strings, use the string version
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  symbolt *isalpha_str_symbol =
    symbol_table_.find_symbol("c:@F@__python_str_isalpha");
  if (!isalpha_str_symbol)
    throw std::runtime_error("str_isalpha function not found in symbol table");

  side_effect_expr_function_callt isalpha_call;
  isalpha_call.function() = symbol_expr(*isalpha_str_symbol);
  isalpha_call.arguments().push_back(str_addr);
  isalpha_call.location() = location;
  isalpha_call.type() = bool_type();

  return isalpha_call;
}

exprt string_handler::handle_string_isspace(
  const exprt &str_expr,
  const locationt &location)
{
  std::string func_symbol_id = ensure_string_function_symbol(
    "__python_str_isspace",
    bool_type(),
    {pointer_typet(char_type())},
    location);

  // Get the string pointer
  exprt str_ptr = str_expr;

  if (str_expr.is_constant() && str_expr.type().is_array())
  {
    str_ptr = exprt("index", pointer_typet(char_type()));
    str_ptr.copy_to_operands(str_expr);
    str_ptr.copy_to_operands(from_integer(0, int_type()));
  }
  else if (str_expr.type().is_array())
  {
    str_ptr = exprt("address_of", pointer_typet(char_type()));
    exprt index_expr("index", char_type());
    index_expr.copy_to_operands(str_expr);
    index_expr.copy_to_operands(from_integer(0, int_type()));
    str_ptr.copy_to_operands(index_expr);
  }
  else if (!str_expr.type().is_pointer())
  {
    str_ptr = exprt("address_of", pointer_typet(char_type()));
    str_ptr.copy_to_operands(str_expr);
  }

  // Create function call
  side_effect_expr_function_callt call;
  call.function() = symbol_exprt(func_symbol_id, code_typet());
  call.arguments().push_back(str_ptr);
  call.type() = bool_type();
  call.location() = location;

  return call;
}

exprt string_handler::handle_char_isspace(
  const exprt &char_expr,
  const locationt &location)
{
  // For single characters, use the standard C isspace() function
  std::string func_symbol_id = ensure_string_function_symbol(
    "isspace", int_type(), {int_type()}, location);

  // Convert char to int for isspace
  exprt char_as_int = char_expr;
  if (char_expr.type() != int_type())
  {
    char_as_int = typecast_exprt(char_expr, int_type());
  }

  // Create function call to C's isspace
  side_effect_expr_function_callt call;
  call.function() = symbol_exprt(func_symbol_id, code_typet());
  call.arguments().push_back(char_as_int);
  call.type() = int_type();
  call.location() = location;

  // Convert result to boolean (isspace returns non-zero for whitespace)
  exprt result("notequal", bool_type());
  result.copy_to_operands(call);
  result.copy_to_operands(from_integer(0, int_type()));

  return result;
}

exprt string_handler::handle_string_lstrip(
  const exprt &str_expr,
  const locationt &location)
{
  std::string func_symbol_id = ensure_string_function_symbol(
    "__python_str_lstrip",
    pointer_typet(char_type()),
    {pointer_typet(char_type())},
    location);

  // Get the string pointer
  exprt str_ptr = str_expr;

  if (str_expr.is_constant() && str_expr.type().is_array())
  {
    str_ptr = exprt("index", pointer_typet(char_type()));
    str_ptr.copy_to_operands(str_expr);
    str_ptr.copy_to_operands(from_integer(0, int_type()));
  }
  else if (str_expr.type().is_array())
  {
    str_ptr = exprt("address_of", pointer_typet(char_type()));
    exprt index_expr("index", char_type());
    index_expr.copy_to_operands(str_expr);
    index_expr.copy_to_operands(from_integer(0, int_type()));
    str_ptr.copy_to_operands(index_expr);
  }
  else if (!str_expr.type().is_pointer())
  {
    str_ptr = exprt("address_of", pointer_typet(char_type()));
    str_ptr.copy_to_operands(str_expr);
  }

  // Create function call
  side_effect_expr_function_callt call;
  call.function() = symbol_exprt(func_symbol_id, code_typet());
  call.arguments().push_back(str_ptr);
  call.type() = pointer_typet(char_type());
  call.location() = location;

  return call;
}

exprt string_handler::handle_string_membership(
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &element)
{
  exprt lhs_char_value =
    python_char_utils::get_char_value_as_int(lhs, true);

  // Use strchr for single character membership testing
  if (!lhs_char_value.is_nil())
  {
    symbolt *strchr_symbol = symbol_table_.find_symbol("c:@F@strchr");
    if (!strchr_symbol)
    {
      // Create strchr symbol if it doesn't exist
      symbolt new_symbol;
      new_symbol.name = "strchr";
      new_symbol.id = "c:@F@strchr";
      new_symbol.mode = "C";
      new_symbol.is_extern = true;

      code_typet strchr_type;
      typet char_ptr = gen_pointer_type(char_type());
      strchr_type.return_type() = char_ptr;
      strchr_type.arguments().push_back(code_typet::argumentt(char_ptr));
      strchr_type.arguments().push_back(code_typet::argumentt(int_type()));
      new_symbol.type = strchr_type;

      symbol_table_.add(new_symbol);
      strchr_symbol = symbol_table_.find_symbol("c:@F@strchr");
    }

    exprt rhs_str = ensure_null_terminated_string(rhs);
    exprt rhs_addr = get_array_base_address(rhs_str);

    exprt char_as_int = typecast_exprt(lhs_char_value, int_type());

    // Call strchr(string, character)
    side_effect_expr_function_callt strchr_call;
    strchr_call.function() = symbol_expr(*strchr_symbol);
    strchr_call.arguments() = {rhs_addr, char_as_int};
    strchr_call.location() = converter_.get_location_from_decl(element);
    strchr_call.type() = gen_pointer_type(char_type());

    // Check if result != NULL (character found)
    constant_exprt null_ptr(gen_pointer_type(char_type()));
    null_ptr.set_value("NULL");

    exprt not_equal("notequal", bool_type());
    not_equal.copy_to_operands(strchr_call, null_ptr);

    return not_equal;
  }

  // Use strstr for substring membership testing
  exprt lhs_str = ensure_null_terminated_string(lhs);
  exprt rhs_str = ensure_null_terminated_string(rhs);

  // Get base addresses for C string functions
  exprt lhs_addr = get_array_base_address(lhs_str);
  exprt rhs_addr = get_array_base_address(rhs_str);

  // Find strstr symbol - returns pointer to first occurrence or NULL
  symbolt *strstr_symbol = symbol_table_.find_symbol("c:@F@strstr");
  if (!strstr_symbol)
    throw std::runtime_error("strstr function not found for 'in' operator");

  // Call strstr(haystack, needle) - in Python "needle in haystack"
  side_effect_expr_function_callt strstr_call;
  strstr_call.function() = symbol_expr(*strstr_symbol);
  strstr_call.arguments() = {
    rhs_addr, lhs_addr}; // haystack is rhs, needle is lhs
  strstr_call.location() = converter_.get_location_from_decl(element);
  strstr_call.type() = gen_pointer_type(char_type());

  // Check if result != NULL (substring found)
  constant_exprt null_ptr(gen_pointer_type(char_type()));
  null_ptr.set_value("NULL");

  exprt not_equal("notequal", bool_type());
  not_equal.copy_to_operands(strstr_call, null_ptr);

  return not_equal;
}

exprt string_handler::handle_string_islower(
  const exprt &string_obj,
  const locationt &location)
{
  // Check if this is a single character
  if (string_obj.type().is_unsignedbv() || string_obj.type().is_signedbv())
  {
    // Call Python's single-character version
    symbolt *islower_symbol =
      symbol_table_.find_symbol("c:@F@__python_char_islower");
    if (!islower_symbol)
      throw std::runtime_error(
        "__python_char_islower function not found in symbol table");

    side_effect_expr_function_callt islower_call;
    islower_call.function() = symbol_expr(*islower_symbol);
    islower_call.arguments().push_back(string_obj);
    islower_call.location() = location;
    islower_call.type() = bool_type();

    return islower_call;
  }

  // For full strings, use the string version
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  symbolt *islower_str_symbol =
    symbol_table_.find_symbol("c:@F@__python_str_islower");
  if (!islower_str_symbol)
    throw std::runtime_error("str_islower function not found in symbol table");

  side_effect_expr_function_callt islower_call;
  islower_call.function() = symbol_expr(*islower_str_symbol);
  islower_call.arguments().push_back(str_addr);
  islower_call.location() = location;
  islower_call.type() = bool_type();

  return islower_call;
}

exprt string_handler::handle_string_lower(
  const exprt &string_obj,
  const locationt &location)
{
  // For single characters, handle directly
  if (string_obj.type().is_unsignedbv() || string_obj.type().is_signedbv())
  {
    symbolt *lower_symbol =
      symbol_table_.find_symbol("c:@F@__python_char_lower");
    if (!lower_symbol)
      throw std::runtime_error(
        "__python_char_lower function not found in symbol table");

    side_effect_expr_function_callt lower_call;
    lower_call.function() = symbol_expr(*lower_symbol);
    lower_call.arguments().push_back(string_obj);
    lower_call.location() = location;
    lower_call.type() = char_type();

    return lower_call;
  }

  // For full strings, use the string version
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  symbolt *lower_str_symbol =
    symbol_table_.find_symbol("c:@F@__python_str_lower");
  if (!lower_str_symbol)
    throw std::runtime_error("str_lower function not found in symbol table");

  side_effect_expr_function_callt lower_call;
  lower_call.function() = symbol_expr(*lower_str_symbol);
  lower_call.arguments().push_back(str_addr);
  lower_call.location() = location;
  lower_call.type() = pointer_typet(char_type());

  return lower_call;
}

exprt string_handler::handle_string_to_int(
  const exprt &string_obj,
  const exprt &base_arg,
  const locationt &location)
{
  // Ensure we have a null-terminated string
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);

  // Get base address of the string
  exprt str_addr = get_array_base_address(str_expr);

  // Determine the base value (default is 10)
  exprt base_expr = base_arg;
  if (base_expr.is_nil())
  {
    // Default base is 10
    base_expr = from_integer(10, int_type());
  }
  else if (!base_expr.type().is_signedbv() && !base_expr.type().is_unsignedbv())
  {
    // Cast base to int if needed
    base_expr = typecast_exprt(base_expr, int_type());
  }

  // Find the __python_int function symbol
  symbolt *int_symbol = symbol_table_.find_symbol("c:@F@__python_int");
  if (!int_symbol)
  {
    throw std::runtime_error("__python_int function not found in symbol table");
  }

  // Call __python_int(str, base)
  side_effect_expr_function_callt int_call;
  int_call.function() = symbol_expr(*int_symbol);
  int_call.arguments().push_back(str_addr);
  int_call.arguments().push_back(base_expr);
  int_call.location() = location;
  int_call.type() = int_type();

  return int_call;
}

exprt string_handler::handle_string_to_int_base10(
  const exprt &string_obj,
  const locationt &location)
{
  // Convenience wrapper for base 10 conversion
  return handle_string_to_int(string_obj, nil_exprt(), location);
}

exprt string_handler::handle_int_conversion(
  const exprt &arg,
  const locationt &location)
{
  // Handle int() with different argument types

  // If argument is already an integer type, return as is
  if (type_utils::is_integer_type(arg.type()))
  {
    return arg;
  }

  // If argument is a float, truncate to integer
  if (arg.type().is_floatbv())
  {
    return typecast_exprt(arg, int_type());
  }

  // If argument is a boolean, convert to 0 or 1
  if (arg.type().is_bool())
  {
    exprt result("if", int_type());
    result.copy_to_operands(arg);
    result.copy_to_operands(from_integer(1, int_type()));
    result.copy_to_operands(from_integer(0, int_type()));
    return result;
  }

  // If argument is a string or char array, use string conversion
  if (arg.type().is_array() && arg.type().subtype() == char_type())
  {
    return handle_string_to_int_base10(arg, location);
  }

  // If argument is a pointer to char (string pointer)
  if (arg.type().is_pointer() && arg.type().subtype() == char_type())
  {
    // Create a wrapper to ensure null-termination handling
    exprt string_copy = arg;
    return handle_string_to_int(string_copy, nil_exprt(), location);
  }

  // For other types, attempt a typecast
  return typecast_exprt(arg, int_type());
}

exprt string_handler::handle_int_conversion_with_base(
  const exprt &arg,
  const exprt &base,
  const locationt &location)
{
  // int() with explicit base only works with strings
  if (!arg.type().is_array() && !arg.type().is_pointer())
  {
    throw std::runtime_error("int() with base argument requires string input");
  }

  // Ensure base is an integer
  exprt base_expr = base;
  if (!base_expr.type().is_signedbv() && !base_expr.type().is_unsignedbv())
  {
    base_expr = typecast_exprt(base_expr, int_type());
  }

  return handle_string_to_int(arg, base_expr, location);
}

exprt string_handler::handle_chr_conversion(
  const exprt &codepoint_arg,
  const locationt &location)
{
  // Ensure the argument is an integer type
  exprt codepoint_expr = codepoint_arg;

  // If not already an integer, try to convert it
  if (!type_utils::is_integer_type(codepoint_expr.type()))
  {
    // If it's a float, truncate to integer
    if (codepoint_expr.type().is_floatbv())
      codepoint_expr = typecast_exprt(codepoint_expr, int_type());
    // If it's a boolean, convert to 0 or 1
    else if (codepoint_expr.type().is_bool())
    {
      exprt result("if", int_type());
      result.copy_to_operands(codepoint_expr);
      result.copy_to_operands(from_integer(1, int_type()));
      result.copy_to_operands(from_integer(0, int_type()));
      codepoint_expr = result;
    }
    else
      throw std::runtime_error("chr() argument must be an integer");
  }

  // Cast to int type if it's a different integer width
  if (codepoint_expr.type() != int_type())
    codepoint_expr = typecast_exprt(codepoint_expr, int_type());

  // Find the __python_chr function symbol
  symbolt *chr_symbol = symbol_table_.find_symbol("c:@F@__python_chr");
  if (!chr_symbol)
    throw std::runtime_error("__python_chr function not found in symbol table");

  // Call __python_chr(codepoint)
  side_effect_expr_function_callt chr_call;
  chr_call.function() = symbol_expr(*chr_symbol);
  chr_call.arguments().push_back(codepoint_expr);
  chr_call.location() = location;
  chr_call.type() = pointer_typet(char_type());

  return chr_call;
}
