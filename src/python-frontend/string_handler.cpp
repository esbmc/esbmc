#include <python-frontend/char_utils.h>
#include <python-frontend/exception_utils.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/python_list.h>
#include <python-frontend/string_handler.h>
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

#include <cmath>
#include <cctype>
#include <cstring>
#include <iomanip>
#include <limits>
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

namespace
{
static bool try_extract_const_string(
  string_handler &handler,
  const exprt &expr,
  std::string &out)
{
  exprt tmp = expr;
  exprt str_expr = handler.ensure_null_terminated_string(tmp);
  if (str_expr.is_symbol() || str_expr.type().is_array())
  {
    out = handler.extract_string_from_array_operands(str_expr);
    return true;
  }
  return false;
}

static bool get_constant_int(const exprt &expr, long long &out)
{
  if (expr.is_nil())
    return false;
  BigInt tmp;
  if (!to_integer(expr, tmp))
    return false;
  out = tmp.to_int64();
  return true;
}

static char to_lower_char(char c)
{
  return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

static char to_upper_char(char c)
{
  return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

static std::string
format_value_from_json(const nlohmann::json &arg, python_converter &converter)
{
  std::string value;
  if (arg.contains("_type") && arg["_type"] == "Constant")
  {
    if (arg["value"].is_null())
      return "None";
    if (arg["value"].is_string())
      return arg["value"].get<std::string>();
    if (arg["value"].is_boolean())
      return arg["value"].get<bool>() ? "True" : "False";
    if (arg["value"].is_number_integer())
      return std::to_string(arg["value"].get<long long>());
    if (arg["value"].is_number_float())
    {
      std::ostringstream oss;
      oss << arg["value"].get<double>();
      return oss.str();
    }
    throw std::runtime_error("format() unsupported constant type");
  }

  if (string_handler::extract_constant_string(arg, converter, value))
    return value;

  throw std::runtime_error("format() requires constant arguments");
}
} // namespace

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

  if (op == "Eq" || op == "NotEq" || type_utils::is_ordered_comparison(op))
    return handle_string_comparison(op, lhs, rhs, element);
  else if (op == "Add")
    return handle_string_concatenation(lhs, rhs, left, right);

  return nil_exprt();
}

exprt string_handler::get_array_base_address(const exprt &arr)
{
  if (
    (arr.is_constant() && arr.type().is_array()) ||
    (arr.id() == "string-constant" && arr.type().is_array()))
  {
    // Avoid taking the address of a constant array directly, which can
    // produce an invalid pointer constant in migration.
    symbolt &tmp = converter_.create_tmp_symbol(
      nlohmann::json(),
      "$str_tmp$",
      arr.type(),
      exprt());
    code_declt decl(symbol_expr(tmp));
    decl.copy_to_operands(arr);
    converter_.add_instruction(decl);
    exprt index = index_exprt(symbol_expr(tmp), from_integer(0, index_type()));
    return address_of_exprt(index);
  }

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
      // Extract index/dereference to avoid nested dereferences
      exprt lhs_value = lhs;
      if (lhs.is_index())
      {
        symbolt &temp = converter_.create_tmp_symbol(
          nlohmann::json(), "$char_temp$", lhs.type(), gen_zero(lhs.type()));
        code_assignt assign(symbol_expr(temp), lhs);
        converter_.add_instruction(assign);
        lhs_value = symbol_expr(temp);
      }

      typet string_type = type_handler_.build_array(char_type(), 2);
      exprt str_array = gen_zero(string_type);
      str_array.operands().at(0) = lhs_value;
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
  const exprt &chars_arg,
  const locationt &location)
{
  bool can_fold_constant = false;
  if (str_expr.type().is_array())
  {
    if (str_expr.is_constant())
      can_fold_constant = true;
    else if (str_expr.is_symbol())
    {
      const symbolt *symbol = symbol_table_.find_symbol(str_expr.identifier());
      if (
        symbol && symbol->value.is_constant() &&
        symbol->value.type().is_array())
        can_fold_constant = true;
    }
  }

  if (chars_arg.is_nil() && string_builder_ && can_fold_constant)
  {
    std::vector<exprt> chars = string_builder_->extract_string_chars(str_expr);
    bool all_constant = true;

    for (const auto &ch : chars)
    {
      if (!ch.is_constant())
      {
        all_constant = false;
        break;
      }
    }

    if (all_constant)
    {
      auto is_whitespace = [](const exprt &ch) -> bool {
        BigInt char_val =
          binary2integer(ch.value().as_string(), ch.type().is_signedbv());
        char c = static_cast<char>(char_val.to_uint64());
        return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' ||
               c == '\r';
      };

      while (!chars.empty() && is_whitespace(chars.front()))
        chars.erase(chars.begin());

      return string_builder_->build_null_terminated_string(chars);
    }
  }

  // If chars_arg is empty, strip whitespace (default behavior)
  std::vector<typet> arg_types = {pointer_typet(char_type())};

  if (chars_arg.is_not_nil())
  {
    // With chars argument: __python_str_lstrip_chars(str, chars)
    arg_types.push_back(pointer_typet(char_type()));

    std::string func_symbol_id = ensure_string_function_symbol(
      "__python_str_lstrip_chars",
      pointer_typet(char_type()),
      arg_types,
      location);

    // Convert arguments to pointers if needed
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

    exprt chars_ptr = chars_arg;
    if (chars_arg.type().is_array())
    {
      chars_ptr = exprt("address_of", pointer_typet(char_type()));
      exprt index_expr("index", char_type());
      index_expr.copy_to_operands(chars_arg);
      index_expr.copy_to_operands(from_integer(0, int_type()));
      chars_ptr.copy_to_operands(index_expr);
    }
    call.arguments().push_back(chars_ptr);

    call.type() = pointer_typet(char_type());
    call.location() = location;

    return call;
  }
  else
  {
    // Without chars argument: __python_str_lstrip(str) - default whitespace
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
}

exprt string_handler::handle_string_strip(
  const exprt &str_expr,
  const exprt &chars_arg,
  const locationt &location)
{
  bool can_fold_constant = false;
  if (str_expr.type().is_array())
  {
    if (str_expr.is_constant())
      can_fold_constant = true;
    else if (str_expr.is_symbol())
    {
      const symbolt *symbol = symbol_table_.find_symbol(str_expr.identifier());
      if (
        symbol && symbol->value.is_constant() &&
        symbol->value.type().is_array())
        can_fold_constant = true;
    }
  }

  if (chars_arg.is_nil() && string_builder_ && can_fold_constant)
  {
    std::vector<exprt> chars = string_builder_->extract_string_chars(str_expr);
    bool all_constant = true;

    for (const auto &ch : chars)
    {
      if (!ch.is_constant())
      {
        all_constant = false;
        break;
      }
    }

    if (all_constant)
    {
      auto is_whitespace = [](const exprt &ch) -> bool {
        BigInt char_val =
          binary2integer(ch.value().as_string(), ch.type().is_signedbv());
        char c = static_cast<char>(char_val.to_uint64());
        return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' ||
               c == '\r';
      };

      while (!chars.empty() && is_whitespace(chars.front()))
        chars.erase(chars.begin());

      while (!chars.empty() && is_whitespace(chars.back()))
        chars.pop_back();

      return string_builder_->build_null_terminated_string(chars);
    }
  }

  // If chars_arg is provided, use __python_str_strip_chars
  if (chars_arg.is_not_nil())
  {
    std::string func_symbol_id = ensure_string_function_symbol(
      "__python_str_strip_chars",
      pointer_typet(char_type()),
      {pointer_typet(char_type()), pointer_typet(char_type())},
      location);

    exprt str_ptr = str_expr;
    if (str_expr.type().is_array())
    {
      str_ptr = exprt("address_of", pointer_typet(char_type()));
      exprt index_expr("index", char_type());
      index_expr.copy_to_operands(str_expr);
      index_expr.copy_to_operands(from_integer(0, int_type()));
      str_ptr.copy_to_operands(index_expr);
    }

    exprt chars_ptr = chars_arg;
    if (chars_arg.type().is_array())
    {
      chars_ptr = exprt("address_of", pointer_typet(char_type()));
      exprt index_expr("index", char_type());
      index_expr.copy_to_operands(chars_arg);
      index_expr.copy_to_operands(from_integer(0, int_type()));
      chars_ptr.copy_to_operands(index_expr);
    }

    side_effect_expr_function_callt call;
    call.function() = symbol_exprt(func_symbol_id, code_typet());
    call.arguments().push_back(str_ptr);
    call.arguments().push_back(chars_ptr);
    call.type() = pointer_typet(char_type());
    call.location() = location;
    return call;
  }

  // Default behavior: strip whitespace
  std::string func_symbol_id = ensure_string_function_symbol(
    "__python_str_strip",
    pointer_typet(char_type()),
    {pointer_typet(char_type())},
    location);

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

  side_effect_expr_function_callt call;
  call.function() = symbol_exprt(func_symbol_id, code_typet());
  call.arguments().push_back(str_ptr);
  call.type() = pointer_typet(char_type());
  call.location() = location;

  return call;
}

exprt string_handler::handle_string_rstrip(
  const exprt &str_expr,
  const exprt &chars_arg,
  const locationt &location)
{
  // If chars_arg is provided, use __python_str_rstrip_chars
  if (chars_arg.is_not_nil())
  {
    std::string func_symbol_id = ensure_string_function_symbol(
      "__python_str_rstrip_chars",
      pointer_typet(char_type()),
      {pointer_typet(char_type()), pointer_typet(char_type())},
      location);

    exprt str_ptr = str_expr;
    if (str_expr.type().is_array())
    {
      str_ptr = exprt("address_of", pointer_typet(char_type()));
      exprt index_expr("index", char_type());
      index_expr.copy_to_operands(str_expr);
      index_expr.copy_to_operands(from_integer(0, int_type()));
      str_ptr.copy_to_operands(index_expr);
    }

    exprt chars_ptr = chars_arg;
    if (chars_arg.type().is_array())
    {
      chars_ptr = exprt("address_of", pointer_typet(char_type()));
      exprt index_expr("index", char_type());
      index_expr.copy_to_operands(chars_arg);
      index_expr.copy_to_operands(from_integer(0, int_type()));
      chars_ptr.copy_to_operands(index_expr);
    }

    side_effect_expr_function_callt call;
    call.function() = symbol_exprt(func_symbol_id, code_typet());
    call.arguments().push_back(str_ptr);
    call.arguments().push_back(chars_ptr);
    call.type() = pointer_typet(char_type());
    call.location() = location;
    return call;
  }

  // Default behavior: strip whitespace (existing code)
  bool can_fold_constant = str_expr.type().is_array();

  if (!can_fold_constant && str_expr.is_symbol())
  {
    const symbolt *symbol = symbol_table_.find_symbol(str_expr.identifier());
    if (symbol && symbol->value.type().is_array())
      can_fold_constant = true;
  }

  if (can_fold_constant && string_builder_)
  {
    std::vector<exprt> chars = string_builder_->extract_string_chars(str_expr);
    bool all_constant = true;

    for (const auto &ch : chars)
    {
      if (!ch.is_constant())
      {
        all_constant = false;
        break;
      }
    }

    if (all_constant)
    {
      auto is_whitespace = [](const exprt &ch) -> bool {
        BigInt char_val =
          binary2integer(ch.value().as_string(), ch.type().is_signedbv());
        char c = static_cast<char>(char_val.to_uint64());
        return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' ||
               c == '\r';
      };

      while (!chars.empty() && is_whitespace(chars.back()))
        chars.pop_back();

      return string_builder_->build_null_terminated_string(chars);
    }
  }

  std::string func_symbol_id = ensure_string_function_symbol(
    "__python_str_rstrip",
    pointer_typet(char_type()),
    {pointer_typet(char_type())},
    location);

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
  bool lhs_is_char_value = false;

  // Get the width of char type from config
  std::size_t char_width = config.ansi_c.char_width;

  // Check if lhs is a symbol holding a character value
  if (lhs.is_symbol())
  {
    const symbolt *sym =
      symbol_table_.find_symbol(lhs.get_string("identifier"));
    if (sym)
    {
      const typet &value_type = sym->value.type();
      if (
        (value_type.is_signedbv() || value_type.is_unsignedbv()) &&
        bv_width(value_type) == char_width)
      {
        lhs_is_char_value = true;
      }
    }
  }

  // Use strchr for single character membership testing
  if (lhs_is_char_value)
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

    // lhs contains the character value (as void*), cast directly to int
    typecast_exprt char_as_int(lhs, int_type());

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

  // Obtain the actual array expression (handle both constants and symbols)
  auto get_array_expr = [this](const exprt &e) -> const exprt * {
    if (e.is_constant() && e.type().is_array())
      return &e;
    if (e.is_symbol())
    {
      const symbolt *sym = symbol_table_.find_symbol(e.identifier());
      if (sym && sym->value.is_constant() && sym->value.type().is_array())
        return &sym->value;
    }
    return nullptr;
  };

  const exprt *needle_array = get_array_expr(lhs_str);
  const exprt *haystack_array = get_array_expr(rhs_str);

  // Special case: empty needle is always found in any haystack (Python semantics)
  if (needle_array && !needle_array->operands().empty())
  {
    const exprt::operandst &needle_ops = needle_array->operands();

    // Check if needle is empty (just the null terminator)
    if (needle_ops.size() == 1 && needle_ops[0].is_constant())
    {
      BigInt first_val = binary2integer(
        needle_ops[0].value().as_string(), needle_ops[0].type().is_signedbv());

      if (first_val == 0)
      {
        // Empty string is always "in" any string in Python
        return gen_boolean(true);
      }
    }
  }

  // Special case: Check if needle starts with '\0' (but is not empty)
  // Python strings with null characters are valid, but we need to handle
  // the C null-terminator semantics vs Python string semantics
  if (needle_array && haystack_array)
  {
    const exprt::operandst &needle_ops = needle_array->operands();
    const exprt::operandst &haystack_ops = haystack_array->operands();

    // Check if needle starts with '\0' and has more than just the terminator
    if (
      needle_ops.size() > 1 && !needle_ops.empty() &&
      needle_ops[0].is_constant())
    {
      BigInt first_val = binary2integer(
        needle_ops[0].value().as_string(), needle_ops[0].type().is_signedbv());

      if (first_val == 0)
      {
        // Needle starts with '\0' but has more characters
        // Check if haystack has any embedded nulls (before the final terminator)
        bool has_embedded_null = false;
        for (size_t i = 0; i + 1 < haystack_ops.size(); ++i)
        {
          if (haystack_ops[i].is_constant())
          {
            BigInt val = binary2integer(
              haystack_ops[i].value().as_string(),
              haystack_ops[i].type().is_signedbv());
            if (val == 0)
            {
              has_embedded_null = true;
              break;
            }
          }
        }

        // If haystack has no embedded nulls, needle starting with '\0' won't be found
        if (!has_embedded_null)
          return gen_boolean(false);

        // Needle is like '\0x' or '\0abc' - need to search for this pattern
        // in haystack that may contain embedded nulls
        bool found = false;
        for (size_t h = 0; h + needle_ops.size() <= haystack_ops.size(); ++h)
        {
          bool match = true;
          for (size_t n = 0; n + 1 < needle_ops.size(); ++n)
          {
            if (
              !haystack_ops[h + n].is_constant() ||
              !needle_ops[n].is_constant())
            {
              match = false;
              break;
            }
            BigInt h_val = binary2integer(
              haystack_ops[h + n].value().as_string(),
              haystack_ops[h + n].type().is_signedbv());
            BigInt n_val = binary2integer(
              needle_ops[n].value().as_string(),
              needle_ops[n].type().is_signedbv());
            if (h_val != n_val)
            {
              match = false;
              break;
            }
          }
          if (match)
          {
            found = true;
            break;
          }
        }
        return gen_boolean(found);
      }
    }
  }

  // TODO: This falls back to C's strstr for non-constant strings,
  // which is unsound if 'lhs' or 'rhs' contain embedded nulls ('\0').
  // For full Python semantics, a null-aware 'strstr' for
  // symbolic/non-constant strings is required.

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

exprt string_handler::handle_string_upper(
  const exprt &string_obj,
  const locationt &location)
{
  // For single characters, handle directly
  if (string_obj.type().is_unsignedbv() || string_obj.type().is_signedbv())
  {
    symbolt *upper_symbol =
      symbol_table_.find_symbol("c:@F@__python_char_upper");
    if (!upper_symbol)
      throw std::runtime_error(
        "__python_char_upper function not found in symbol table");

    side_effect_expr_function_callt upper_call;
    upper_call.function() = symbol_expr(*upper_symbol);
    upper_call.arguments().push_back(string_obj);
    upper_call.location() = location;
    upper_call.type() = char_type();

    return upper_call;
  }

  // For full strings, use the string version
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  symbolt *upper_str_symbol =
    symbol_table_.find_symbol("c:@F@__python_str_upper");
  if (!upper_str_symbol)
    throw std::runtime_error("str_upper function not found in symbol table");

  side_effect_expr_function_callt upper_call;
  upper_call.function() = symbol_expr(*upper_str_symbol);
  upper_call.arguments().push_back(str_addr);
  upper_call.location() = location;
  upper_call.type() = pointer_typet(char_type());

  return upper_call;
}

exprt string_handler::handle_string_find(
  const exprt &string_obj,
  const exprt &find_arg,
  const locationt &location)
{
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  exprt arg_copy = find_arg;
  exprt arg_expr = ensure_null_terminated_string(arg_copy);
  exprt arg_addr = get_array_base_address(arg_expr);

  symbolt *find_str_symbol =
    symbol_table_.find_symbol("c:@F@__python_str_find");
  if (!find_str_symbol)
    throw std::runtime_error("str_find function not found in symbol table");

  side_effect_expr_function_callt find_call;
  find_call.function() = symbol_expr(*find_str_symbol);
  find_call.arguments().push_back(str_addr);
  find_call.arguments().push_back(arg_addr);
  find_call.location() = location;
  find_call.type() = int_type();

  return find_call;
}

exprt string_handler::handle_string_find_range(
  const exprt &string_obj,
  const exprt &find_arg,
  const exprt &start_arg,
  const exprt &end_arg,
  const locationt &location)
{
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  exprt arg_copy = find_arg;
  exprt arg_expr = ensure_null_terminated_string(arg_copy);
  exprt arg_addr = get_array_base_address(arg_expr);

  exprt start_expr = start_arg;
  if (start_expr.type() != int_type())
    start_expr = typecast_exprt(start_expr, int_type());

  exprt end_expr = end_arg;
  if (end_expr.type() != int_type())
    end_expr = typecast_exprt(end_expr, int_type());

  symbolt *find_range_symbol =
    symbol_table_.find_symbol("c:@F@__python_str_find_range");
  if (!find_range_symbol)
    throw std::runtime_error(
      "str_find_range function not found in symbol table");

  side_effect_expr_function_callt find_call;
  find_call.function() = symbol_expr(*find_range_symbol);
  find_call.arguments().push_back(str_addr);
  find_call.arguments().push_back(arg_addr);
  find_call.arguments().push_back(start_expr);
  find_call.arguments().push_back(end_expr);
  find_call.location() = location;
  find_call.type() = int_type();

  return find_call;
}

exprt string_handler::handle_string_index(
  const nlohmann::json &call,
  const exprt &string_obj,
  const exprt &find_arg,
  const locationt &location)
{
  exprt find_expr = handle_string_find(string_obj, find_arg, location);
  return build_string_index_result(call, find_expr, location);
}

exprt string_handler::handle_string_index_range(
  const nlohmann::json &call,
  const exprt &string_obj,
  const exprt &find_arg,
  const exprt &start_arg,
  const exprt &end_arg,
  const locationt &location)
{
  exprt find_expr = handle_string_find_range(
    string_obj, find_arg, start_arg, end_arg, location);
  return build_string_index_result(call, find_expr, location);
}

exprt string_handler::build_string_index_result(
  const nlohmann::json &call,
  const exprt &find_expr,
  const locationt &location)
{
  symbolt &find_result = converter_.create_tmp_symbol(
    call, "$str_index$", int_type(), gen_zero(int_type()));
  code_declt decl(symbol_expr(find_result));
  decl.location() = location;
  converter_.add_instruction(decl);

  code_assignt assign(symbol_expr(find_result), find_expr);
  assign.location() = location;
  converter_.add_instruction(assign);

  exprt not_found =
    equality_exprt(symbol_expr(find_result), from_integer(-1, int_type()));
  exprt raise = python_exception_utils::make_exception_raise(
    type_handler_, "ValueError", "substring not found", &location);

  code_expressiont raise_code(raise);
  raise_code.location() = location;

  code_ifthenelset if_stmt;
  if_stmt.cond() = not_found;
  if_stmt.then_case() = raise_code;
  if_stmt.location() = location;
  converter_.add_instruction(if_stmt);

  return symbol_expr(find_result);
}

exprt string_handler::handle_string_rfind(
  const exprt &string_obj,
  const exprt &find_arg,
  const locationt &location)
{
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  exprt arg_copy = find_arg;
  exprt arg_expr = ensure_null_terminated_string(arg_copy);
  exprt arg_addr = get_array_base_address(arg_expr);

  symbolt *rfind_str_symbol =
    symbol_table_.find_symbol("c:@F@__python_str_rfind");
  if (!rfind_str_symbol)
    throw std::runtime_error("str_rfind function not found in symbol table");

  side_effect_expr_function_callt rfind_call;
  rfind_call.function() = symbol_expr(*rfind_str_symbol);
  rfind_call.arguments().push_back(str_addr);
  rfind_call.arguments().push_back(arg_addr);
  rfind_call.location() = location;
  rfind_call.type() = int_type();

  return rfind_call;
}

exprt string_handler::handle_string_rfind_range(
  const exprt &string_obj,
  const exprt &find_arg,
  const exprt &start_arg,
  const exprt &end_arg,
  const locationt &location)
{
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  exprt arg_copy = find_arg;
  exprt arg_expr = ensure_null_terminated_string(arg_copy);
  exprt arg_addr = get_array_base_address(arg_expr);

  exprt start_expr = start_arg;
  if (start_expr.type() != int_type())
    start_expr = typecast_exprt(start_expr, int_type());

  exprt end_expr = end_arg;
  if (end_expr.type() != int_type())
    end_expr = typecast_exprt(end_expr, int_type());

  symbolt *rfind_range_symbol =
    symbol_table_.find_symbol("c:@F@__python_str_rfind_range");
  if (!rfind_range_symbol)
    throw std::runtime_error(
      "str_rfind_range function not found in symbol table");

  side_effect_expr_function_callt rfind_call;
  rfind_call.function() = symbol_expr(*rfind_range_symbol);
  rfind_call.arguments().push_back(str_addr);
  rfind_call.arguments().push_back(arg_addr);
  rfind_call.arguments().push_back(start_expr);
  rfind_call.arguments().push_back(end_expr);
  rfind_call.location() = location;
  rfind_call.type() = int_type();

  return rfind_call;
}

exprt string_handler::handle_string_replace(
  const exprt &string_obj,
  const exprt &old_arg,
  const exprt &new_arg,
  const exprt &count_arg,
  const locationt &location)
{
  // Try to handle constant strings directly using string_builder API
  // This avoids the loop unwinding issues in ESBMC

  exprt string_copy = string_obj;
  exprt old_copy = old_arg;
  exprt new_copy = new_arg;

  // Ensure all are proper strings
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt old_expr = ensure_null_terminated_string(old_copy);
  exprt new_expr = ensure_null_terminated_string(new_copy);

  // Extract the count value
  int64_t max_replacements = -1; // -1 means replace all
  if (count_arg.is_constant())
  {
    BigInt count_val = binary2integer(
      count_arg.value().as_string(), count_arg.type().is_signedbv());
    max_replacements = count_val.to_int64();

    // count=0 means no replacements
    if (max_replacements == 0)
      return str_expr;
  }

  // Try to extract constant string values for compile-time replacement
  std::string src_str = extract_string_from_array_operands(str_expr);
  std::string old_str = extract_string_from_array_operands(old_expr);
  std::string new_str = extract_string_from_array_operands(new_expr);

  // If we can extract all strings as constants, do replacement at compile time
  if (!src_str.empty() || str_expr.operands().size() > 0)
  {
    // Handle the case where source string might be empty but valid
    if (str_expr.type().is_array() && str_expr.operands().size() > 0)
    {
      src_str = extract_string_from_array_operands(str_expr);
    }

    // Perform the replacement
    std::string result;
    int64_t replacements_done = 0;

    if (old_str.empty())
    {
      // Special case: empty old string - insert new_str between each char
      for (size_t i = 0; i < src_str.size(); ++i)
      {
        if (max_replacements < 0 || replacements_done < max_replacements)
        {
          result += new_str;
          replacements_done++;
        }
        result += src_str[i];
      }
      // Add new_str at the end if we still have replacements left
      if (max_replacements < 0 || replacements_done < max_replacements)
      {
        result += new_str;
      }
    }
    else
    {
      // Normal replacement
      size_t pos = 0;
      size_t old_len = old_str.length();

      while (pos < src_str.length())
      {
        size_t found = src_str.find(old_str, pos);

        if (
          found == std::string::npos ||
          (max_replacements >= 0 && replacements_done >= max_replacements))
        {
          // No more matches or reached max replacements - copy rest
          result += src_str.substr(pos);
          break;
        }

        // Copy characters before the match
        result += src_str.substr(pos, found - pos);
        // Add replacement string
        result += new_str;
        replacements_done++;
        // Move past the matched substring
        pos = found + old_len;
      }
    }

    // Build the result string using string_builder
    return string_builder_->build_string_literal(result);
  }

  // Fallback to C function for non-constant strings
  exprt str_addr =
    str_expr.type().is_pointer() ? str_expr : get_array_base_address(str_expr);
  exprt old_addr =
    old_expr.type().is_pointer() ? old_expr : get_array_base_address(old_expr);
  exprt new_addr =
    new_expr.type().is_pointer() ? new_expr : get_array_base_address(new_expr);

  std::string func_symbol_id = ensure_string_function_symbol(
    "__python_str_replace",
    pointer_typet(char_type()),
    {pointer_typet(char_type()),
     pointer_typet(char_type()),
     pointer_typet(char_type()),
     int_type()},
    location);

  side_effect_expr_function_callt replace_call;
  replace_call.function() = symbol_exprt(func_symbol_id, code_typet());
  replace_call.arguments() = {str_addr, old_addr, new_addr, count_arg};
  replace_call.location() = location;
  replace_call.type() = pointer_typet(char_type());

  return replace_call;
}

exprt string_handler::handle_string_capitalize(
  const exprt &string_obj,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string(*this, string_obj, input))
    throw std::runtime_error("capitalize() requires constant string");

  if (!input.empty())
  {
    input[0] = to_upper_char(input[0]);
    for (size_t i = 1; i < input.size(); ++i)
      input[i] = to_lower_char(input[i]);
  }

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for capitalize()");

  exprt result = string_builder_->build_string_literal(input);
  result.location() = location;
  return result;
}

exprt string_handler::handle_string_title(
  const exprt &string_obj,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string(*this, string_obj, input))
    throw std::runtime_error("title() requires constant string");

  bool new_word = true;
  for (char &ch : input)
  {
    if (std::isalpha(static_cast<unsigned char>(ch)))
    {
      ch = new_word ? to_upper_char(ch) : to_lower_char(ch);
      new_word = false;
    }
    else
    {
      new_word = !std::isalnum(static_cast<unsigned char>(ch));
    }
  }

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for title()");

  exprt result = string_builder_->build_string_literal(input);
  result.location() = location;
  return result;
}

exprt string_handler::handle_string_swapcase(
  const exprt &string_obj,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string(*this, string_obj, input))
    throw std::runtime_error("swapcase() requires constant string");

  for (char &ch : input)
  {
    unsigned char uch = static_cast<unsigned char>(ch);
    if (std::islower(uch))
      ch = to_upper_char(ch);
    else if (std::isupper(uch))
      ch = to_lower_char(ch);
  }

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for swapcase()");

  exprt result = string_builder_->build_string_literal(input);
  result.location() = location;
  return result;
}

exprt string_handler::handle_string_casefold(
  const exprt &string_obj,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string(*this, string_obj, input))
    throw std::runtime_error("casefold() requires constant string");

  for (char &ch : input)
    ch = to_lower_char(ch);

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for casefold()");

  exprt result = string_builder_->build_string_literal(input);
  result.location() = location;
  return result;
}

exprt string_handler::handle_string_count(
  const exprt &string_obj,
  const exprt &sub_arg,
  const exprt &start_arg,
  const exprt &end_arg,
  const locationt &location)
{
  std::string input;
  std::string sub;
  if (
    !try_extract_const_string(*this, string_obj, input) ||
    !try_extract_const_string(*this, sub_arg, sub))
  {
    throw std::runtime_error("count() requires constant strings");
  }

  long long start = 0;
  long long end = static_cast<long long>(input.size());
  long long tmp = 0;
  if (get_constant_int(start_arg, tmp))
    start = tmp;
  if (get_constant_int(end_arg, tmp))
    end = tmp;

  if (start < 0)
    start = static_cast<long long>(input.size()) + start;
  if (end < 0)
    end = static_cast<long long>(input.size()) + end;

  if (start < 0)
    start = 0;
  if (end < 0)
    end = 0;
  if (start > static_cast<long long>(input.size()))
    start = static_cast<long long>(input.size());
  if (end > static_cast<long long>(input.size()))
    end = static_cast<long long>(input.size());
  if (end < start)
    end = start;

  long long count = 0;
  if (sub.empty())
  {
    count = (end - start) + 1;
  }
  else
  {
    size_t pos = static_cast<size_t>(start);
    size_t limit = static_cast<size_t>(end);
    while (pos <= limit && pos + sub.size() <= limit)
    {
      size_t found = input.find(sub, pos);
      if (found == std::string::npos || found + sub.size() > limit)
        break;
      ++count;
      pos = found + sub.size();
    }
  }

  exprt result = from_integer(count, long_long_int_type());
  result.location() = location;
  return result;
}

exprt string_handler::handle_string_removeprefix(
  const exprt &string_obj,
  const exprt &prefix_arg,
  const locationt &location)
{
  std::string input;
  std::string prefix;
  if (
    !try_extract_const_string(*this, string_obj, input) ||
    !try_extract_const_string(*this, prefix_arg, prefix))
  {
    throw std::runtime_error("removeprefix() requires constant strings");
  }

  if (!prefix.empty() && input.rfind(prefix, 0) == 0)
    input = input.substr(prefix.size());

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for removeprefix()");

  exprt result = string_builder_->build_string_literal(input);
  result.location() = location;
  return result;
}

exprt string_handler::handle_string_removesuffix(
  const exprt &string_obj,
  const exprt &suffix_arg,
  const locationt &location)
{
  std::string input;
  std::string suffix;
  if (
    !try_extract_const_string(*this, string_obj, input) ||
    !try_extract_const_string(*this, suffix_arg, suffix))
  {
    throw std::runtime_error("removesuffix() requires constant strings");
  }

  if (
    !suffix.empty() && input.size() >= suffix.size() &&
    input.compare(input.size() - suffix.size(), suffix.size(), suffix) == 0)
  {
    input.resize(input.size() - suffix.size());
  }

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for removesuffix()");

  exprt result = string_builder_->build_string_literal(input);
  result.location() = location;
  return result;
}

exprt string_handler::handle_string_splitlines(
  const nlohmann::json &call,
  const exprt &string_obj,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string(*this, string_obj, input))
    throw std::runtime_error("splitlines() requires constant string");

  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t i = 0; i < input.size(); ++i)
  {
    if (input[i] == '\n' || input[i] == '\r')
    {
      parts.push_back(input.substr(start, i - start));
      if (input[i] == '\r' && (i + 1) < input.size() && input[i + 1] == '\n')
        ++i;
      start = i + 1;
    }
  }
  if (start < input.size())
    parts.push_back(input.substr(start));

  nlohmann::json list_node;
  list_node["_type"] = "List";
  list_node["elts"] = nlohmann::json::array();
  converter_.copy_location_fields_from_decl(call, list_node);

  for (const auto &part : parts)
  {
    nlohmann::json elem;
    elem["_type"] = "Constant";
    elem["value"] = part;
    converter_.copy_location_fields_from_decl(call, elem);
    list_node["elts"].push_back(elem);
  }

  python_list list(converter_, list_node);
  exprt result = list.get();
  result.location() = location;
  return result;
}

exprt string_handler::handle_string_format(
  const nlohmann::json &call,
  const exprt &string_obj,
  const locationt &location)
{
  std::string format_str;
  if (!try_extract_const_string(*this, string_obj, format_str))
    throw std::runtime_error("format() requires constant format string");

  std::vector<std::string> args;
  if (call.contains("args") && call["args"].is_array())
  {
    for (const auto &arg : call["args"])
    {
      args.push_back(format_value_from_json(arg, converter_));
    }
  }

  std::unordered_map<std::string, std::string> keywords;
  if (call.contains("keywords") && call["keywords"].is_array())
  {
    for (const auto &kw : call["keywords"])
    {
      if (!kw.contains("arg") || kw["arg"].is_null())
        throw std::runtime_error("format() kwargs are not supported");
      std::string key = kw["arg"].get<std::string>();
      if (!kw.contains("value"))
        throw std::runtime_error("format() keyword missing value");
      keywords.emplace(key, format_value_from_json(kw["value"], converter_));
    }
  }

  std::string result;
  result.reserve(format_str.size());
  size_t arg_index = 0;

  for (size_t i = 0; i < format_str.size();)
  {
    char ch = format_str[i];
    if (ch == '{')
    {
      if ((i + 1) < format_str.size() && format_str[i + 1] == '{')
      {
        result.push_back('{');
        i += 2;
        continue;
      }

      size_t end = format_str.find('}', i + 1);
      if (end == std::string::npos)
        throw std::runtime_error("format() unmatched '{'");

      std::string field = format_str.substr(i + 1, end - (i + 1));
      if (field.empty())
      {
        if (arg_index >= args.size())
          throw std::runtime_error("format() missing arguments");
        result += args[arg_index++];
      }
      else
      {
        bool all_digits = true;
        for (char fc : field)
        {
          if (!std::isdigit(static_cast<unsigned char>(fc)))
          {
            all_digits = false;
            break;
          }
        }

        if (all_digits)
        {
          size_t idx = static_cast<size_t>(std::stoull(field));
          if (idx >= args.size())
            throw std::runtime_error("format() argument index out of range");
          result += args[idx];
        }
        else
        {
          auto it = keywords.find(field);
          if (it == keywords.end())
            throw std::runtime_error("format() missing keyword argument");
          result += it->second;
        }
      }

      i = end + 1;
      continue;
    }
    if (ch == '}')
    {
      if ((i + 1) < format_str.size() && format_str[i + 1] == '}')
      {
        result.push_back('}');
        i += 2;
        continue;
      }
      throw std::runtime_error("format() unmatched '}'");
    }

    result.push_back(ch);
    ++i;
  }

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for format()");

  exprt out = string_builder_->build_string_literal(result);
  out.location() = location;
  return out;
}

exprt string_handler::handle_string_partition(
  const exprt &string_obj,
  const exprt &sep_arg,
  const locationt &location)
{
  std::string input;
  std::string sep;
  if (
    !try_extract_const_string(*this, string_obj, input) ||
    !try_extract_const_string(*this, sep_arg, sep))
  {
    throw std::runtime_error("partition() requires constant strings");
  }
  if (sep.empty())
    throw std::runtime_error("partition() separator cannot be empty");

  std::string before;
  std::string after;
  std::string mid;
  size_t pos = input.find(sep);
  if (pos == std::string::npos)
  {
    before = input;
    mid = "";
    after = "";
  }
  else
  {
    before = input.substr(0, pos);
    mid = sep;
    after = input.substr(pos + sep.size());
  }

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for partition()");

  exprt before_expr = string_builder_->build_string_literal(before);
  exprt mid_expr = string_builder_->build_string_literal(mid);
  exprt after_expr = string_builder_->build_string_literal(after);

  struct_typet tuple_type;
  tuple_type.components().push_back(
    struct_typet::componentt("element_0", before_expr.type()));
  tuple_type.components().push_back(
    struct_typet::componentt("element_1", mid_expr.type()));
  tuple_type.components().push_back(
    struct_typet::componentt("element_2", after_expr.type()));

  struct_exprt tuple_expr(tuple_type);
  tuple_expr.operands() = {before_expr, mid_expr, after_expr};
  tuple_expr.location() = location;
  return tuple_expr;
}

exprt string_handler::handle_string_isalnum(
  const exprt &string_obj,
  [[maybe_unused]] const locationt &location)
{
  std::string input;
  if (!try_extract_const_string(*this, string_obj, input))
    throw std::runtime_error("isalnum() requires constant string");
  if (input.empty())
    return from_integer(0, bool_type());

  for (char ch : input)
  {
    if (!std::isalnum(static_cast<unsigned char>(ch)))
      return from_integer(0, bool_type());
  }
  return from_integer(1, bool_type());
}

exprt string_handler::handle_string_isupper(
  const exprt &string_obj,
  [[maybe_unused]] const locationt &location)
{
  std::string input;
  if (!try_extract_const_string(*this, string_obj, input))
    throw std::runtime_error("isupper() requires constant string");
  bool has_cased = false;
  for (char ch : input)
  {
    unsigned char uch = static_cast<unsigned char>(ch);
    if (std::islower(uch))
      return from_integer(0, bool_type());
    if (std::isupper(uch))
      has_cased = true;
  }
  return from_integer(has_cased ? 1 : 0, bool_type());
}

exprt string_handler::handle_string_isnumeric(
  const exprt &string_obj,
  [[maybe_unused]] const locationt &location)
{
  std::string input;
  if (!try_extract_const_string(*this, string_obj, input))
    throw std::runtime_error("isnumeric() requires constant string");
  if (input.empty())
    return from_integer(0, bool_type());

  for (char ch : input)
  {
    if (!std::isdigit(static_cast<unsigned char>(ch)))
      return from_integer(0, bool_type());
  }
  return from_integer(1, bool_type());
}

exprt string_handler::handle_string_isidentifier(
  const exprt &string_obj,
  [[maybe_unused]] const locationt &location)
{
  std::string input;
  if (!try_extract_const_string(*this, string_obj, input))
    throw std::runtime_error("isidentifier() requires constant string");
  if (input.empty())
    return from_integer(0, bool_type());

  unsigned char first = static_cast<unsigned char>(input[0]);
  if (!(std::isalpha(first) || input[0] == '_'))
    return from_integer(0, bool_type());
  for (size_t i = 1; i < input.size(); ++i)
  {
    unsigned char ch = static_cast<unsigned char>(input[i]);
    if (!(std::isalnum(ch) || input[i] == '_'))
      return from_integer(0, bool_type());
  }
  return from_integer(1, bool_type());
}

exprt string_handler::handle_string_center(
  const exprt &string_obj,
  const exprt &width_arg,
  const exprt &fill_arg,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string(*this, string_obj, input))
    throw std::runtime_error("center() requires constant string");
  if (!string_builder_)
    throw std::runtime_error("string_builder not set for center()");

  long long width = 0;
  if (!get_constant_int(width_arg, width))
    throw std::runtime_error("center() requires constant width");

  char fill = ' ';
  if (!fill_arg.is_nil())
  {
    std::string fill_str;
    if (
      !try_extract_const_string(*this, fill_arg, fill_str) ||
      fill_str.size() != 1)
    {
      throw std::runtime_error("center() fillchar must be a single character");
    }
    fill = fill_str[0];
  }

  if (width <= static_cast<long long>(input.size()))
    return string_builder_->build_string_literal(input);

  long long pad = width - static_cast<long long>(input.size());
  long long left = pad / 2;
  long long right = pad - left;
  std::string result(static_cast<size_t>(left), fill);
  result += input;
  result.append(static_cast<size_t>(right), fill);

  exprt out = string_builder_->build_string_literal(result);
  out.location() = location;
  return out;
}

exprt string_handler::handle_string_ljust(
  const exprt &string_obj,
  const exprt &width_arg,
  const exprt &fill_arg,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string(*this, string_obj, input))
    throw std::runtime_error("ljust() requires constant string");
  if (!string_builder_)
    throw std::runtime_error("string_builder not set for ljust()");

  long long width = 0;
  if (!get_constant_int(width_arg, width))
    throw std::runtime_error("ljust() requires constant width");

  char fill = ' ';
  if (!fill_arg.is_nil())
  {
    std::string fill_str;
    if (
      !try_extract_const_string(*this, fill_arg, fill_str) ||
      fill_str.size() != 1)
    {
      throw std::runtime_error("ljust() fillchar must be a single character");
    }
    fill = fill_str[0];
  }

  if (width <= static_cast<long long>(input.size()))
    return string_builder_->build_string_literal(input);

  std::string result = input;
  result.append(static_cast<size_t>(width - input.size()), fill);
  exprt out = string_builder_->build_string_literal(result);
  out.location() = location;
  return out;
}

exprt string_handler::handle_string_rjust(
  const exprt &string_obj,
  const exprt &width_arg,
  const exprt &fill_arg,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string(*this, string_obj, input))
    throw std::runtime_error("rjust() requires constant string");
  if (!string_builder_)
    throw std::runtime_error("string_builder not set for rjust()");

  long long width = 0;
  if (!get_constant_int(width_arg, width))
    throw std::runtime_error("rjust() requires constant width");

  char fill = ' ';
  if (!fill_arg.is_nil())
  {
    std::string fill_str;
    if (
      !try_extract_const_string(*this, fill_arg, fill_str) ||
      fill_str.size() != 1)
    {
      throw std::runtime_error("rjust() fillchar must be a single character");
    }
    fill = fill_str[0];
  }

  if (width <= static_cast<long long>(input.size()))
    return string_builder_->build_string_literal(input);

  std::string result(static_cast<size_t>(width - input.size()), fill);
  result += input;
  exprt out = string_builder_->build_string_literal(result);
  out.location() = location;
  return out;
}

exprt string_handler::handle_string_zfill(
  const exprt &string_obj,
  const exprt &width_arg,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string(*this, string_obj, input))
    throw std::runtime_error("zfill() requires constant string");
  if (!string_builder_)
    throw std::runtime_error("string_builder not set for zfill()");

  long long width = 0;
  if (!get_constant_int(width_arg, width))
    throw std::runtime_error("zfill() requires constant width");

  if (width <= static_cast<long long>(input.size()))
    return string_builder_->build_string_literal(input);

  size_t pad = static_cast<size_t>(width - input.size());
  std::string result;
  if (!input.empty() && (input[0] == '+' || input[0] == '-'))
  {
    result.push_back(input[0]);
    result.append(pad, '0');
    result.append(input.substr(1));
  }
  else
  {
    result.append(pad, '0');
    result.append(input);
  }

  exprt out = string_builder_->build_string_literal(result);
  out.location() = location;
  return out;
}

exprt string_handler::handle_string_expandtabs(
  const exprt &string_obj,
  const exprt &tabsize_arg,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string(*this, string_obj, input))
    throw std::runtime_error("expandtabs() requires constant string");
  if (!string_builder_)
    throw std::runtime_error("string_builder not set for expandtabs()");

  long long tabsize = 8;
  long long tmp = 0;
  if (get_constant_int(tabsize_arg, tmp))
    tabsize = tmp;
  if (tabsize < 0)
    tabsize = 0;

  std::string result;
  size_t column = 0;
  for (char ch : input)
  {
    if (ch == '\t')
    {
      size_t spaces =
        tabsize == 0 ? 0 : (tabsize - (column % static_cast<size_t>(tabsize)));
      result.append(spaces, ' ');
      column += spaces;
    }
    else
    {
      result.push_back(ch);
      if (ch == '\n' || ch == '\r')
        column = 0;
      else
        ++column;
    }
  }
  exprt out = string_builder_->build_string_literal(result);
  out.location() = location;
  return out;
}

exprt string_handler::handle_string_format_map(
  const nlohmann::json &call,
  const exprt &string_obj,
  const locationt &location)
{
  if (!call.contains("args") || call["args"].size() != 1)
    throw std::runtime_error("format_map() requires one argument");

  std::string format_str;
  if (!try_extract_const_string(*this, string_obj, format_str))
    throw std::runtime_error("format_map() requires constant format string");

  const auto &mapping = call["args"][0];
  if (!mapping.contains("_type") || mapping["_type"] != "Dict")
    throw std::runtime_error("format_map() requires constant dict");

  std::unordered_map<std::string, std::string> values;
  const auto &keys = mapping["keys"];
  const auto &vals = mapping["values"];
  for (size_t i = 0; i < keys.size(); ++i)
  {
    std::string key;
    if (!extract_constant_string(keys[i], converter_, key))
      throw std::runtime_error("format_map() keys must be constant strings");
    values.emplace(key, format_value_from_json(vals[i], converter_));
  }

  std::string result;
  result.reserve(format_str.size());

  for (size_t i = 0; i < format_str.size();)
  {
    char ch = format_str[i];
    if (ch == '{')
    {
      if ((i + 1) < format_str.size() && format_str[i + 1] == '{')
      {
        result.push_back('{');
        i += 2;
        continue;
      }

      size_t end = format_str.find('}', i + 1);
      if (end == std::string::npos)
        throw std::runtime_error("format_map() unmatched '{'");

      std::string field = format_str.substr(i + 1, end - (i + 1));
      if (field.empty())
        throw std::runtime_error("format_map() requires named fields");

      auto it = values.find(field);
      if (it == values.end())
        throw std::runtime_error("format_map() missing key");
      result += it->second;
      i = end + 1;
      continue;
    }
    if (ch == '}')
    {
      if ((i + 1) < format_str.size() && format_str[i + 1] == '}')
      {
        result.push_back('}');
        i += 2;
        continue;
      }
      throw std::runtime_error("format_map() unmatched '}'");
    }

    result.push_back(ch);
    ++i;
  }

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for format_map()");
  exprt out = string_builder_->build_string_literal(result);
  out.location() = location;
  return out;
}

bool string_handler::extract_constant_string(
  const nlohmann::json &node,
  python_converter &converter,
  std::string &out)
{
  if (
    node.contains("_type") && node["_type"] == "Constant" &&
    node.contains("value") && node["value"].is_string())
  {
    out = node["value"].get<std::string>();
    return true;
  }

  if (node.contains("_type") && node["_type"] == "Name" && node.contains("id"))
  {
    const std::string var_name = node["id"].get<std::string>();
    nlohmann::json var_value = json_utils::get_var_value(
      var_name, converter.get_current_func_name(), converter.get_ast_json());

    if (
      !var_value.empty() && var_value.contains("value") &&
      var_value["value"].contains("_type") &&
      var_value["value"]["_type"] == "Constant" &&
      var_value["value"].contains("value") &&
      var_value["value"]["value"].is_string())
    {
      out = var_value["value"]["value"].get<std::string>();
      return true;
    }
  }

  return false;
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

exprt string_handler::handle_str_join(const nlohmann::json &call_json)
{
  // Validate JSON structure: ensure we have the required keys
  if (!call_json.contains("args") || call_json["args"].empty())
    throw std::runtime_error("join() missing required argument: 'iterable'");

  if (!call_json.contains("func"))
    throw std::runtime_error("invalid join() call");

  const auto &func = call_json["func"];

  // Verify this is an Attribute call (method call syntax: obj.method())
  // and has the value (the separator object)
  if (
    !func.contains("_type") || func["_type"] != "Attribute" ||
    !func.contains("value"))
    throw std::runtime_error("invalid join() call");

  // Extract separator: for " ".join(l), func["value"] is the Constant " "
  exprt separator = converter_.get_expr(func["value"]);
  ensure_string_array(separator);

  // Get the list argument (the iterable to join)
  const nlohmann::json &list_arg = call_json["args"][0];

  auto can_fold_elements = [](const nlohmann::json &elements) -> bool {
    if (!elements.is_array())
      return false;
    for (const auto &elem : elements)
    {
      if (
        !elem.contains("_type") || elem["_type"] != "Constant" ||
        !elem.contains("value") || !elem["value"].is_string())
        return false;
    }
    return true;
  };

  auto emit_type_error = [&](const std::string &msg) -> exprt {
    locationt location = converter_.get_location_from_decl(call_json);
    code_assertt assert_code(gen_boolean(false));
    assert_code.location() = location;
    assert_code.location().user_provided(true);
    assert_code.location().comment(msg);
    converter_.current_block->copy_to_operands(assert_code);
    return string_builder_->build_string_literal("");
  };

  auto fold_join = [&](const nlohmann::json &elements) -> exprt {
    // Edge case: empty list returns empty string
    if (elements.empty())
    {
      // Create a proper null-terminated empty string
      typet empty_string_type = type_handler_.build_array(char_type(), 1);
      exprt empty_str = gen_zero(empty_string_type);
      // Explicitly set the first (and only) element to null terminator
      empty_str.operands().at(0) = from_integer(0, char_type());
      return empty_str;
    }

    // Convert JSON elements to ESBMC expressions
    std::vector<exprt> elem_exprs;
    for (const auto &elem : elements)
    {
      exprt elem_expr = converter_.get_expr(elem);
      ensure_string_array(elem_expr);
      elem_exprs.push_back(elem_expr);
    }

    // Edge case: single element returns the element itself (no separator)
    if (elem_exprs.size() == 1)
      return elem_exprs[0];

    // Main algorithm: Build the joined string by extracting characters
    // from all elements and separators, then constructing a single string.
    // This avoids multiple concatenation operations which could cause
    // null terminator issues.
    std::vector<exprt> all_chars;

    // Start with the first element
    std::vector<exprt> first_chars =
      string_builder_->extract_string_chars(elem_exprs[0]);
    all_chars.insert(all_chars.end(), first_chars.begin(), first_chars.end());

    // For each remaining element: add separator, then add element
    for (size_t i = 1; i < elem_exprs.size(); ++i)
    {
      // Insert separator characters
      std::vector<exprt> sep_chars =
        string_builder_->extract_string_chars(separator);
      all_chars.insert(all_chars.end(), sep_chars.begin(), sep_chars.end());

      // Insert element characters
      std::vector<exprt> elem_chars =
        string_builder_->extract_string_chars(elem_exprs[i]);
      all_chars.insert(all_chars.end(), elem_chars.begin(), elem_chars.end());
    }

    // Build final null-terminated string from all collected characters
    return string_builder_->build_null_terminated_string(all_chars);
  };

  // Constant-fold for direct List literals of constant strings.
  if (list_arg.contains("_type") && list_arg["_type"] == "List")
  {
    if (list_arg.contains("elts") && can_fold_elements(list_arg["elts"]))
    {
      const auto &elements = list_arg["elts"];
      return fold_join(elements);
    }

    if (list_arg.contains("elts") && list_arg["elts"].is_array())
      return emit_type_error("join() requires a list of strings");
  }

  // Constant-fold for variables initialized with List literals of constants.
  if (
    list_arg.contains("_type") && list_arg["_type"] == "Name" &&
    list_arg.contains("id"))
  {
    std::string var_name = list_arg["id"].get<std::string>();

    nlohmann::json var_decl = json_utils::find_var_decl(
      var_name, converter_.get_current_func_name(), converter_.get_ast_json());

    if (
      !var_decl.empty() && var_decl.contains("value") &&
      var_decl["value"].contains("_type") &&
      var_decl["value"]["_type"] == "List")
    {
      if (
        var_decl["value"].contains("elts") &&
        can_fold_elements(var_decl["value"]["elts"]))
      {
        return fold_join(var_decl["value"]["elts"]);
      }

      if (var_decl["value"].contains("elts") &&
          var_decl["value"]["elts"].is_array())
        return emit_type_error("join() requires a list of strings");
    }
  }

  // Runtime join for list variables (e.g., "".join(list_var))
  if (
    list_arg.contains("_type") && list_arg["_type"] == "Name" &&
    list_arg.contains("id"))
  {
    const std::string func_name = "c:@F@__python_str_join";
    const symbolt *func_symbol = symbol_table_.find_symbol(func_name);

    if (!func_symbol)
    {
      // Create function type: char* __python_str_join(PyListObject* list, const char* sep)
      code_typet func_type;
      func_type.return_type() = pointer_typet(char_type());

      code_typet::argumentt list_arg_type;
      list_arg_type.type() = type_handler_.get_list_type();
      func_type.arguments().push_back(list_arg_type);

      code_typet::argumentt sep_arg_type;
      sep_arg_type.type() = pointer_typet(char_type());
      func_type.arguments().push_back(sep_arg_type);

      symbolt new_symbol;
      new_symbol.name = func_name;
      new_symbol.id = func_name;
      new_symbol.type = func_type;
      new_symbol.mode = "C";
      new_symbol.module = "python";
      new_symbol.location = converter_.get_location_from_decl(call_json);
      new_symbol.is_extern = true;

      converter_.add_symbol(new_symbol);
      func_symbol = symbol_table_.find_symbol(func_name);
    }

    exprt list_expr = converter_.get_expr(list_arg);
    exprt sep_expr = separator;
    if (sep_expr.type().is_array())
      sep_expr = get_array_base_address(sep_expr);

    side_effect_expr_function_callt join_call;
    join_call.function() = symbol_expr(*func_symbol);
    join_call.arguments().push_back(list_expr);
    join_call.arguments().push_back(sep_expr);
    join_call.type() = pointer_typet(char_type());
    join_call.location() = converter_.get_location_from_decl(call_json);
    return join_call;
  }

  throw std::runtime_error("join() argument must be a list of strings");
}

exprt string_handler::create_char_comparison_expr(
  const std::string &op,
  const exprt &lhs_char_value,
  const exprt &rhs_char_value,
  const exprt &lhs_source,
  const exprt &rhs_source) const
{
  // Create comparison expression with integer operands
  exprt comp_expr(converter_.get_op(op, bool_type()), bool_type());
  comp_expr.copy_to_operands(lhs_char_value, rhs_char_value);

  // Preserve location from original operands
  if (!lhs_source.location().is_nil())
    comp_expr.location() = lhs_source.location();
  else if (!rhs_source.location().is_nil())
    comp_expr.location() = rhs_source.location();

  return comp_expr;
}

exprt string_handler::handle_single_char_comparison(
  const std::string &op,
  exprt &lhs,
  exprt &rhs)
{
  // Dereference pointer to character if needed
  auto maybe_dereference = [](const exprt &expr) -> exprt {
    if (
      expr.type().is_pointer() && (expr.type().subtype().is_signedbv() ||
                                   expr.type().subtype().is_unsignedbv()))
    {
      exprt deref("dereference", expr.type().subtype());
      deref.copy_to_operands(expr);
      return deref;
    }
    return expr;
  };

  // Create comparison expression with location info
  auto create_comparison = [&](const exprt &left, const exprt &right) -> exprt {
    exprt comp_expr(converter_.get_op(op, bool_type()), bool_type());
    comp_expr.copy_to_operands(left, right);

    if (!lhs.location().is_nil())
      comp_expr.location() = lhs.location();
    else if (!rhs.location().is_nil())
      comp_expr.location() = rhs.location();

    return comp_expr;
  };

  exprt lhs_to_check = maybe_dereference(lhs);
  exprt rhs_to_check = maybe_dereference(rhs);

  // Try to get character values from the (potentially dereferenced) expressions
  exprt lhs_char_value =
    python_char_utils::get_char_value_as_int(lhs_to_check, false);
  exprt rhs_char_value =
    python_char_utils::get_char_value_as_int(rhs_to_check, false);

  // If both are valid character values, do the comparison
  if (!lhs_char_value.is_nil() && !rhs_char_value.is_nil())
    return create_char_comparison_expr(
      op, lhs_char_value, rhs_char_value, lhs, rhs);

  // Handle mixed cases: dereferenced pointer with valid character value
  if (lhs_to_check.id() == "dereference" && !rhs_char_value.is_nil())
  {
    exprt lhs_as_int = typecast_exprt(lhs_to_check, rhs_char_value.type());
    return create_comparison(lhs_as_int, rhs_char_value);
  }

  if (!lhs_char_value.is_nil() && rhs_to_check.id() == "dereference")
  {
    exprt rhs_as_int = typecast_exprt(rhs_to_check, lhs_char_value.type());
    return create_comparison(lhs_char_value, rhs_as_int);
  }

  return nil_exprt();
}
