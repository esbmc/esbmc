#pragma once

#include <util/expr.h>
#include <util/type.h>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <cstdint>

// Forward declaration
class python_converter;
class type_handler;
class contextt;
class string_handler;

/// Helper class for building and manipulating string expressions in Python frontend
/// Handles null-termination, character extraction, and string concatenation
class string_builder
{
public:
  explicit string_builder(python_converter &converter, string_handler *handler);

  /// Create a null terminator character constant
  exprt make_null_terminator();

  /// Create a character constant from a byte value
  exprt make_char_constant(unsigned char ch);

  /// Extract characters from an expression, stopping at null terminator
  /// Returns a vector of character expressions (without null terminator)
  std::vector<exprt> extract_string_chars(
    const exprt &expr,
    const nlohmann::json &json_node = nlohmann::json());

  /// Create a null-terminated string array from a vector of character expressions
  exprt build_null_terminated_string(const std::vector<exprt> &chars);

  /// Create a null-terminated string from a std::string
  exprt build_string_literal(const std::string &str);

  /// Create a null-terminated string from a vector of bytes
  exprt build_byte_string(const std::vector<uint8_t> &bytes);

  /// Ensure an expression is a null-terminated string
  /// Converts single characters or promotes to proper string arrays
  exprt ensure_null_terminated_string(exprt &e);

  /// Concatenate two string expressions with proper null-termination
  exprt concatenate_strings(
    const exprt &lhs,
    const exprt &rhs,
    const nlohmann::json &left = nlohmann::json(),
    const nlohmann::json &right = nlohmann::json());

  /// Create a raw byte array without null termination (for Python bytes literals)
  exprt build_raw_byte_array(const std::vector<uint8_t> &bytes);

private:
  python_converter &converter_;
  string_handler *str_handler_;
  contextt &get_symbol_table() const;
  type_handler &get_type_handler();
};
