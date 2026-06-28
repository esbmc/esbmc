#include <python-frontend/function_call/expr.h>
#include <python-frontend/json_utils.h>
#include <irep2/irep2_utils.h>
#include <python-frontend/python_exception_handler.h>
#include <python-frontend/string/string_builder.h>
#include <python-frontend/string/string_handler.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/message.h>
#include <util/migrate.h>
#include <util/std_expr.h>

#include <algorithm>
#include <cctype>
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

    exprt val = sym->get_value();

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

// Fold a known code point into an integer expression, rewriting the AST node
// in place so converter_.get_expr() builds an int constant.
exprt function_call_expr::build_ord_constant(
  nlohmann::json &arg,
  int code_point) const
{
  arg["_type"] = "Constant";
  arg.erase("id");
  arg.erase("ctx");
  arg["value"] = code_point;
  arg["type"] = "int";

  exprt expr = converter_.get_expr(arg);
  expr.type() = type_handler_.get_typet("int", 0);
  return expr;
}

exprt function_call_expr::handle_ord(nlohmann::json &arg) const
{
  // Fast path: constant string literal. Folding also handles multi-byte UTF-8
  // code points that the byte-level runtime string model cannot reconstruct.
  if (is_string_arg(arg))
    return build_ord_constant(
      arg, decode_utf8_codepoint(arg["value"].get<std::string>()));

  // A Name bound to a constant string: fold from the stored value. Gate on the
  // value being a string so a non-string variable (e.g. `a: str = 1`) skips
  // folding and falls through to the runtime tail, which raises TypeError.
  if (arg["_type"] == "Name" && arg.contains("id"))
  {
    const symbolt *sym = lookup_python_symbol(arg["id"]);
    if (!sym)
      return converter_.get_exception_handler().gen_exception_raise(
        "NameError",
        "variable '" + arg["id"].get<std::string>() + "' is not defined");

    if (type_utils::is_string_type(sym->get_value().type()))
      if (auto value_opt = extract_string_from_symbol(sym))
        return build_ord_constant(arg, decode_utf8_codepoint(*value_opt));
    // Non-constant string variable: fall through to runtime evaluation.
  }

  // Runtime evaluation for slices, method results, and string variables without
  // a constant value: ord(s), ord(s[i]), ord(s.lower()).
  exprt expr = converter_.get_expr(arg);

  // Propagate an exception raised while evaluating the argument.
  if (expr.statement() == "cpp-throw")
    return expr;

  // A character from string indexing is an 8-bit int tagged #cpp_type==char.
  // V.3: build the char->int cast in IREP2, back-migrating once (mirrors the
  // build_typecast helper; typecast2t round-trips byte-identically).
  if (type_utils::is_char_type(expr.type()))
  {
    expr2tc expr2;
    migrate_expr(expr, expr2);
    return migrate_expr_back(typecast2tc(migrate_type(int_type()), expr2));
  }

  // A runtime string: return the code point of its first character.
  if (type_utils::is_string_type(expr.type()))
    return converter_.get_string_handler().handle_ord_conversion(
      expr, converter_.get_location_from_decl(call_));

  return converter_.get_exception_handler().gen_exception_raise(
    "TypeError", "ord() expected string of length 1");
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

exprt function_call_expr::handle_base_conversion(
  nlohmann::json &arg,
  const std::string &func_name,
  const std::string &prefix,
  std::ios_base &(*base_formatter)(std::ios_base &)) const
{
  long long int_value = 0;
  bool is_negative = false;

  // Extract integer value from argument
  if (arg.contains("_type") && arg["_type"] == "UnaryOp")
  {
    const auto &op = arg["op"];
    const auto &operand = arg["operand"];

    if (
      op["_type"] == "USub" && operand.contains("value") &&
      operand["value"].is_number_integer())
    {
      int_value = operand["value"].get<long long>();

      // Treat -0 as 0 (consistent with Python behavior)
      if (int_value != 0)
        is_negative = true;
    }
    else
    {
      return converter_.get_exception_handler().gen_exception_raise(
        "TypeError", "Unsupported UnaryOp in " + func_name + "()");
    }
  }
  else if (arg.contains("value") && arg["value"].is_number_integer())
  {
    int_value = arg["value"].get<long long>();
    if (int_value < 0)
      is_negative = true;
  }
  else
  {
    // Not an AST integer literal. Try evaluating the argument:
    //   - If it constant-folds to an integer (e.g. `bin(round(3.0))` where
    //     handle_round's compile-time numeric path produces a `Constant`),
    //     drive the existing string-building path below.
    //   - Otherwise the value is symbolic; lower to the matching runtime
    //     operational model (__python_int_to_{bin,hex,oct}) so the result
    //     depends on the actual runtime int rather than being unreachable.
    try
    {
      exprt operand_expr = converter_.get_expr(arg);
      if (!type_utils::is_integer_type(operand_expr.type()))
        return converter_.get_exception_handler().gen_exception_raise(
          "TypeError", func_name + "() argument must be an integer");

      expr2tc operand2;
      migrate_expr(operand_expr, operand2);
      simplify(operand2);
      BigInt extracted;
      if (!to_integer(operand2, extracted))
      {
        int_value = static_cast<long long>(extracted.to_int64());
        if (int_value < 0)
          is_negative = true;
      }
      else
      {
        // Symbolic int — dispatch to the runtime OM matching this builtin.
        // Feed the simplified operand, migrated back to the legacy exprt the
        // string builder expects.
        operand_expr = migrate_expr_back(operand2);
        const std::string fn_name = func_name == "bin" ? "__python_int_to_bin"
                                    : func_name == "hex"
                                      ? "__python_int_to_hex"
                                      : "__python_int_to_oct";
        return converter_.get_string_builder()
          .build_runtime_str_conversion_call(
            fn_name, long_long_int_type(), operand_expr);
      }
    }
    catch (const std::exception &)
    {
      return converter_.get_exception_handler().gen_exception_raise(
        "TypeError", func_name + "() argument must be an integer");
    }
  }

  std::string result_str;
  if (func_name == "bin")
  {
    // C++ iostreams have no binary formatter, so build the digits manually.
    // Compute the magnitude via unsigned modular negation (well-defined for
    // LLONG_MIN, unlike std::llabs). Key off int_value's own sign: the USub
    // branch above stores a positive magnitude with is_negative set, while the
    // direct-integer branch leaves int_value signed.
    unsigned long long mag =
      (int_value < 0) ? (0ULL - static_cast<unsigned long long>(int_value))
                      : static_cast<unsigned long long>(int_value);
    std::string digits;
    if (mag == 0)
      digits = "0";
    else
      while (mag != 0)
      {
        digits.push_back(static_cast<char>('0' + (mag & 1U)));
        mag >>= 1U;
      }
    std::reverse(digits.begin(), digits.end());
    result_str = (is_negative ? "-" + prefix : prefix) + digits;
  }
  else
  {
    // Convert to string with appropriate base and prefix
    std::ostringstream oss;
    oss << (is_negative ? "-" + prefix : prefix) << base_formatter;

    // For hex, also apply nouppercase
    if (func_name == "hex")
      oss << std::nouppercase;

    oss << std::llabs(int_value);
    result_str = oss.str();
  }

  // Create string type and return character array expression
  typet t = type_handler_.get_typet("str", result_str.size() + 1);
  std::vector<uint8_t> string_literal(result_str.begin(), result_str.end());
  return converter_.make_char_array_expr(string_literal, t);
}

exprt function_call_expr::handle_hex(nlohmann::json &arg) const
{
  return handle_base_conversion(arg, "hex", "0x", std::hex);
}

exprt function_call_expr::handle_oct(nlohmann::json &arg) const
{
  return handle_base_conversion(arg, "oct", "0o", std::oct);
}

exprt function_call_expr::handle_ascii() const
{
  const auto &args = call_["args"];
  if (args.size() != 1)
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError", "ascii() takes exactly one argument");

  const auto &arg = args[0];

  // String constant: produce the quoted, ASCII-escaped repr form. This is the
  // only case where ascii() differs from str(): non-printable and non-ASCII
  // code points are rendered as \xNN / \uNNNN / \UNNNNNNNN escapes.
  if (
    arg["_type"] == "Constant" && arg.contains("value") &&
    arg["value"].is_string())
  {
    const std::string s = arg["value"].get<std::string>();

    // CPython prefers single quotes, switching to double quotes only when the
    // string contains ' but not ".
    const char quote =
      (s.find('\'') != std::string::npos && s.find('"') == std::string::npos)
        ? '"'
        : '\'';

    // cp is a Unicode code point (<= 0x10FFFF), so an unsigned int holds it
    // and the widest escape is \U + 8 hex digits; using %08x (not %08lx) keeps
    // the output within buf and avoids GCC's -Wformat-truncation.
    auto append_escape = [](std::string &out, unsigned cp) {
      char buf[12];
      if (cp <= 0xff)
        std::snprintf(buf, sizeof(buf), "\\x%02x", cp);
      else if (cp <= 0xffff)
        std::snprintf(buf, sizeof(buf), "\\u%04x", cp);
      else
        std::snprintf(buf, sizeof(buf), "\\U%08x", cp);
      out += buf;
    };

    std::string out;
    out.push_back(quote);
    for (size_t i = 0; i < s.size();)
    {
      const unsigned char c = static_cast<unsigned char>(s[i]);
      if (c == static_cast<unsigned char>(quote) || c == '\\')
      {
        out.push_back('\\');
        out.push_back(static_cast<char>(c));
        ++i;
      }
      else if (c == '\n')
      {
        out += "\\n";
        ++i;
      }
      else if (c == '\r')
      {
        out += "\\r";
        ++i;
      }
      else if (c == '\t')
      {
        out += "\\t";
        ++i;
      }
      else if (c < 0x20 || c == 0x7f)
      {
        append_escape(out, c);
        ++i;
      }
      else if (c < 0x80)
      {
        out.push_back(static_cast<char>(c));
        ++i;
      }
      else
      {
        // Decode one UTF-8 sequence to a code point, then escape it.
        unsigned cp = 0;
        int extra = 0;
        if ((c & 0xe0) == 0xc0)
        {
          cp = c & 0x1f;
          extra = 1;
        }
        else if ((c & 0xf0) == 0xe0)
        {
          cp = c & 0x0f;
          extra = 2;
        }
        else if ((c & 0xf8) == 0xf0)
        {
          cp = c & 0x07;
          extra = 3;
        }
        else
        {
          // Invalid lead byte: escape the raw byte.
          append_escape(out, c);
          ++i;
          continue;
        }
        if (i + extra >= s.size())
        {
          append_escape(out, c);
          ++i;
          continue;
        }
        for (int k = 1; k <= extra; ++k)
          cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3f);
        append_escape(out, cp);
        i += extra + 1;
      }
    }
    out.push_back(quote);
    return converter_.get_string_builder().build_string_literal(out);
  }

  // Non-string argument: ascii(x) == repr(x), which for numbers and bools is
  // identical to str(x). Reuse the existing str conversion machinery.
  exprt value_expr = converter_.get_expr(arg);
  if (!value_expr.is_nil() && value_expr.statement() != "cpp-throw")
  {
    const typet &vt = value_expr.type();
    if (vt.is_bool() || type_utils::is_integer_type(vt) || vt.is_floatbv())
      return converter_.get_string_handler().convert_to_string(value_expr);
  }
  return converter_.get_exception_handler().gen_exception_raise(
    "TypeError", "ascii() argument type not supported");
}

exprt function_call_expr::handle_format() const
{
  const auto &args = call_["args"];
  if (args.empty() || args.size() > 2)
    throw std::runtime_error("format() takes one or two arguments");

  // The format spec (default "") must be a constant string. Only a bare
  // presentation-type spec is folded; any width/alignment/precision spec
  // (e.g. "08x", ">10") is left unsupported rather than mis-folded.
  std::string spec;
  if (args.size() == 2)
  {
    if (!args[1].contains("value") || !args[1]["value"].is_string())
      throw std::runtime_error(
        "format() currently requires a constant string spec");
    spec = args[1]["value"].get<std::string>();
  }
  if (spec.size() > 1)
    throw std::runtime_error(
      "format() spec '" + spec + "' is not supported");
  const char type = spec.empty() ? '\0' : spec[0];

  // Integer value with a base spec ('d'/''/'x'/'X'/'o'/'b'). A bool is an int
  // subclass in Python, but a JSON bool is not is_number_integer, so True/False
  // fall through to the unsupported path below (rare for format()).
  //
  // Only a genuine literal value is folded — a Constant or a unary +/- over one.
  // extract_constant_integer would also resolve a Name to its bound constant,
  // but that value can be stale after a reassignment (`x = 255; x = 10`), which
  // would mis-fold; a variable argument is left unsupported instead.
  const std::string value_node_type = args[0].value("_type", std::string());
  const bool is_literal_value =
    value_node_type == "Constant" || value_node_type == "UnaryOp";
  long long v = 0;
  const bool is_int =
    is_literal_value &&
    json_utils::extract_constant_integer(
      args[0], converter_.get_current_func_name(), converter_.get_ast_json(), v);
  if (
    is_int && (type == '\0' || type == 'd' || type == 'x' || type == 'X' ||
               type == 'o' || type == 'b'))
  {
    const bool negative = v < 0;
    // Magnitude in the unsigned domain so LLONG_MIN does not overflow.
    unsigned long long mag =
      negative ? 0ULL - static_cast<unsigned long long>(v)
               : static_cast<unsigned long long>(v);

    std::string digits;
    if (type == '\0' || type == 'd')
      digits = std::to_string(mag);
    else
    {
      const unsigned base = (type == 'o') ? 8 : (type == 'b') ? 2 : 16;
      const char *alphabet =
        (type == 'X') ? "0123456789ABCDEF" : "0123456789abcdef";
      if (mag == 0)
        digits = "0";
      for (; mag != 0; mag /= base)
        digits.insert(digits.begin(), alphabet[mag % base]);
    }
    if (negative)
      digits.insert(digits.begin(), '-');
    return converter_.get_string_builder().build_string_literal(digits);
  }

  // format(value) / format(value, "") on a constant string is the string
  // itself (the default str.__format__ with an empty spec).
  if (
    type == '\0' && args[0].contains("value") &&
    args[0]["value"].is_string())
    return converter_.get_string_builder().build_string_literal(
      args[0]["value"].get<std::string>());

  throw std::runtime_error(
    "format() currently supports a constant int (with a base spec) or a "
    "constant str");
}

exprt function_call_expr::handle_bin(nlohmann::json &arg) const
{
  // The formatter is unused for binary (handled manually in
  // handle_base_conversion); std::dec is passed only to satisfy the signature.
  return handle_base_conversion(arg, "bin", "0b", std::dec);
}

/// Extracts the character string represented by a symbol's constant value.
std::optional<std::string>
function_call_expr::extract_string_from_symbol(const symbolt *sym) const
{
  const exprt &val = sym->get_value();
  std::string result;

  if (val.id() == "if" && val.operands().size() == 3)
  {
    const exprt &cond = val.operands()[0];

    symbolt true_sym;
    true_sym.set_value(val.operands()[1]);
    true_sym.set_type(true_sym.get_value().type());
    auto true_text = extract_string_from_symbol(&true_sym);

    symbolt false_sym;
    false_sym.set_value(val.operands()[2]);
    false_sym.set_type(false_sym.get_value().type());
    auto false_text = extract_string_from_symbol(&false_sym);

    if (cond.is_true())
      return true_text;
    if (cond.is_false())
      return false_text;

    if (true_text && false_text && *true_text == *false_text)
      return true_text;
    return std::nullopt;
  }

  std::function<std::optional<char>(const exprt &)> decode_char =
    [&](const exprt &expr) -> std::optional<char> {
    try
    {
      if (expr.id() == "typecast" && !expr.operands().empty())
        return decode_char(expr.operands().front());

      if (expr.id() == "if" && expr.operands().size() == 3)
      {
        const exprt &cond = expr.operands()[0];
        const exprt &true_case = expr.operands()[1];
        const exprt &false_case = expr.operands()[2];

        if (cond.is_true())
          return decode_char(true_case);
        if (cond.is_false())
          return decode_char(false_case);

        auto true_char = decode_char(true_case);
        auto false_char = decode_char(false_case);
        if (true_char && false_char && *true_char == *false_char)
          return true_char;
        return std::nullopt;
      }

      if (!expr.is_constant())
        return std::nullopt;

      const auto &const_expr = to_constant_expr(expr);
      std::string binary_str = id2string(const_expr.get_value());
      unsigned c = std::stoul(binary_str, nullptr, 2);
      return static_cast<char>(c);
    }
    catch (const std::exception &e)
    {
      log_error("Failed to decode character: {}", e.what());
      return std::nullopt;
    }
  };

  if (val.type().is_array() && val.has_operands())
  {
    for (const auto &ch : val.operands())
    {
      auto decoded = decode_char(ch);
      if (!decoded)
        return std::nullopt;
      if (*decoded == '\0')
        break;
      result += *decoded;
    }
  }
  else if (val.is_constant() && val.type().is_signedbv())
  {
    auto decoded = decode_char(val);
    if (!decoded)
      return std::nullopt;
    result += *decoded;
  }
  else
  {
    log_error("Unhandled symbol format in string extraction.");
    return std::nullopt;
  }

  return result;
}

exprt function_call_expr::handle_str_symbol_to_float(const symbolt *sym) const
{
  auto value_opt = extract_string_from_symbol(sym);
  if (!value_opt)
    return from_double(0.0, type_handler_.get_typet("float", 0));

  {
    char *end = nullptr;
    double dval = std::strtod(value_opt->c_str(), &end);
    if (!end || end != value_opt->c_str() + value_opt->size())
    {
      log_error(
        "Failed float conversion from string \"{}\": invalid argument",
        *value_opt);
      return from_double(0.0, type_handler_.get_typet("float", 0));
    }
    return from_double(dval, type_handler_.get_typet("float", 0));
  }
}

exprt function_call_expr::handle_str_symbol_to_int(const symbolt *sym) const
{
  auto value_opt = extract_string_from_symbol(sym);
  if (!value_opt)
    return from_integer(0, type_handler_.get_typet("int", 0));

  const std::string &value = *value_opt;
  if (value.empty() || !std::all_of(value.begin(), value.end(), ::isdigit))
  {
    log_error("Invalid string for integer conversion: \"{}\"", value);
    return from_integer(0, type_handler_.get_typet("int", 0));
  }

  try
  {
    int int_val = std::stoi(value);
    return from_integer(int_val, type_handler_.get_typet("int", 0));
  }
  catch (const std::exception &e)
  {
    log_error("Failed int conversion from string \"{}\": {}", value, e.what());
    return from_integer(0, type_handler_.get_typet("int", 0));
  }
}
