#ifndef PYTHON_FRONTEND_STRING_HANDLER_H
#define PYTHON_FRONTEND_STRING_HANDLER_H

#include <util/expr.h>
#include <util/std_types.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/context.h>
#include <util/message.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// Forward declarations
class python_converter;
class type_handler;
class string_builder;

/**
 * @brief Handles all string-related operations for Python-to-C conversion
 * 
 * This class extracts string manipulation functionality from python_converter
 * to improve modularity and maintainability. It handles:
 * - String size calculations
 * - String-to-expression conversions
 * - String formatting and f-string processing
 * - String comparison operations
 * - String method operations (startswith, endswith, isdigit, isalpha, etc.)
 * - String concatenation and membership testing
 * - String joining operations (str.join)
 * - Character-level comparisons
 */
class string_handler
{
public:
  /**
   * @brief Constructs a string_handler
   * @param converter Reference to the parent python_converter
   * @param symbol_table Reference to the symbol table (contextt)
   * @param type_handler Reference to the type handler
   * @param str_builder Pointer to the string builder
   */
  string_handler(
    python_converter &converter,
    contextt &symbol_table,
    type_handler &type_handler,
    string_builder *str_builder);

  void set_string_builder(string_builder *sb)
  {
    string_builder_ = sb;
  }

  // String size and conversion operations

  /**
   * @brief Calculate the size of a string expression
   * @param expr Expression to measure
   * @return Size of string including null terminator
   */
  BigInt get_string_size(const exprt &expr);

  /**
   * @brief Convert an expression to a string representation
   * @param expr Expression to convert
   * @return String array expression
   */
  exprt convert_to_string(const exprt &expr);

  /**
   * @brief Extract string content from array operands
   * @param array_expr Array expression containing characters
   * @return Extracted string
   */
  std::string extract_string_from_array_operands(const exprt &array_expr) const;

  /**
   * @brief Ensure an expression is a null-terminated string array
   * @param expr Expression to convert
   */
  void ensure_string_array(exprt &expr);

  /**
   * @brief Ensure string is null-terminated
   * @param e String expression
   * @return Null-terminated string expression
   */
  exprt ensure_null_terminated_string(exprt &e);

  // Format string operations

  /**
   * @brief Process format specification for f-strings
   * @param format_spec JSON format specification
   * @return Format string
   */
  std::string process_format_spec(const nlohmann::json &format_spec);

  /**
   * @brief Apply format specification to an expression
   * @param expr Expression to format
   * @param format Format string
   * @return Formatted expression
   */
  exprt
  apply_format_specification(const exprt &expr, const std::string &format);

  /**
   * @brief Convert f-string JSON to expression
   * @param element JSON element representing f-string
   * @return Concatenated f-string expression
   */
  exprt get_fstring_expr(const nlohmann::json &element);

  // String concatenation operations

  /**
   * @brief Handle string concatenation
   * @param lhs Left operand
   * @param rhs Right operand
   * @param left JSON node for left operand
   * @param right JSON node for right operand
   * @return Concatenated string expression
   */
  exprt handle_string_concatenation(
    const exprt &lhs,
    const exprt &rhs,
    const nlohmann::json &left,
    const nlohmann::json &right);

  /**
   * @brief Handle string concatenation with type promotion
   * @param lhs Left operand
   * @param rhs Right operand
   * @param left JSON node for left operand
   * @param right JSON node for right operand
   * @return Concatenated string expression
   */
  exprt handle_string_concatenation_with_promotion(
    exprt &lhs,
    exprt &rhs,
    const nlohmann::json &left,
    const nlohmann::json &right);

  // String comparison operations

  /**
   * @brief Handle string comparison operations
   * @param op Comparison operator (Eq, NotEq, etc.)
   * @param lhs Left operand
   * @param rhs Right operand
   * @param element JSON element with location info
   * @return Comparison expression or nil_exprt to continue with standard comparison
   */
  exprt handle_string_comparison(
    const std::string &op,
    exprt &lhs,
    exprt &rhs,
    const nlohmann::json &element);

  /**
   * @brief Handle single character comparisons
   * @param op Comparison operator
   * @param lhs Left operand
   * @param rhs Right operand
   * @return Comparison expression or nil_exprt if not a char comparison
   */
  exprt
  handle_single_char_comparison(const std::string &op, exprt &lhs, exprt &rhs);

  /**
   * @brief Create a character comparison expression
   * @param op Comparison operator
   * @param lhs_char_value Left character value (as integer)
   * @param rhs_char_value Right character value (as integer)
   * @param lhs_source Original left source expression
   * @param rhs_source Original right source expression
   * @return Comparison expression with proper location info
   */
  exprt create_char_comparison_expr(
    const std::string &op,
    const exprt &lhs_char_value,
    const exprt &rhs_char_value,
    const exprt &lhs_source,
    const exprt &rhs_source) const;

  /**
   * @brief Handle string repetition
   * @param op multiply operator (Eq, Mult)
   * @param lhs Left operand
   * @param rhs Right operand
   * @param element JSON element with location info
   * @return Repetition expression
   */
  exprt handle_string_repetition(exprt &lhs, exprt &rhs);

  /**
   * @brief Handle general string operations
   * @param op Operation type
   * @param lhs Left operand
   * @param rhs Right operand
   * @param left JSON node for left operand
   * @param right JSON node for right operand
   * @param element JSON element with location info
   * @return Operation result expression
   */
  exprt handle_string_operations(
    const std::string &op,
    exprt &lhs,
    exprt &rhs,
    const nlohmann::json &left,
    const nlohmann::json &right,
    const nlohmann::json &element);

  // String joining operations

  /**
   * @brief Handle str.join() method
   * @param call_json JSON node representing the join call
   * @return Expression representing the joined string
   * 
   * Handles Python's str.join(iterable) method, e.g.:
   * - " ".join(["a", "b", "c"]) -> "a b c"
   * - "-".join(list_var) -> joined string
   */
  exprt handle_str_join(const nlohmann::json &call_json);

  // String method operations

  /**
   * @brief Handle str.startswith() method
   * @param string_obj String object
   * @param prefix_arg Prefix to check
   * @param location Source location
   * @return Boolean expression result
   */
  exprt handle_string_startswith(
    const exprt &string_obj,
    const exprt &prefix_arg,
    const locationt &location);

  /**
   * @brief Handle str.endswith() method
   * @param string_obj String object
   * @param suffix_arg Suffix to check
   * @param location Source location
   * @return Boolean expression result
   */
  exprt handle_string_endswith(
    const exprt &string_obj,
    const exprt &suffix_arg,
    const locationt &location);

  /**
   * @brief Handle str.isdigit() method
   * @param string_obj String object
   * @param location Source location
   * @return Boolean expression result
   */
  exprt
  handle_string_isdigit(const exprt &string_obj, const locationt &location);

  /**
   * @brief Handle str.isalpha() method
   * @param string_obj String object (can be string or single char)
   * @param location Source location
   * @return Boolean expression result
   */
  exprt
  handle_string_isalpha(const exprt &string_obj, const locationt &location);

  /**
   * @brief Handle str.isspace() method for strings
   * @param str_expr String expression
   * @param location Source location
   * @return Boolean expression result
   */
  exprt handle_string_isspace(const exprt &str_expr, const locationt &location);

  /**
   * @brief Handle char.isspace() method for single characters
   * @param char_expr Character expression
   * @param location Source location
   * @return Boolean expression result
   */
  exprt handle_char_isspace(const exprt &char_expr, const locationt &location);

  /**
   * @brief Handle str.lstrip() method
   * @param str_expr String expression
   * @param location Source location
   * @return Pointer to stripped string
   */
  exprt handle_string_lstrip(const exprt &str_expr, const locationt &location);

  /**
   * @brief Handle str.strip() method
   * @param str_expr String expression
   * @param location Source location
   * @return Pointer to stripped string
   */
  exprt handle_string_strip(const exprt &str_expr, const locationt &location);

  /**
   * @brief Handle 'in' operator for strings
   * @param lhs Substring to find
   * @param rhs String to search in
   * @param element JSON element with location info
   * @return Boolean expression result
   */
  exprt handle_string_membership(
    exprt &lhs,
    exprt &rhs,
    const nlohmann::json &element);

  /**
   * @brief Handle Python's str.islower() method
   * @param string_obj Expression representing the string or character to check
   * @param location Source location for error reporting
   * @return Boolean expression: true if all cased chars are lowercase, false otherwise
   */
  exprt
  handle_string_islower(const exprt &string_obj, const locationt &location);

  /**
   * @brief Handle str.lower() method
   * @param string_obj String object
   * @param location Source location
   * @return Pointer to lowercase string
   */
  exprt handle_string_lower(const exprt &string_obj, const locationt &location);

  /**
   * @brief Handle str.find() method
   * @param string_obj String object
   * @param find_arg String to check
   * @param location Source location
   * @return returns the index of the first occurrence of the substring.
   * If not found, it returns -1.
   */
  exprt handle_string_find(
    const exprt &string_obj,
    const exprt &find_arg,
    const locationt &location);

  // Utility methods

  /**
   * @brief Check if an expression is a zero-length array
   * @param expr Expression to check
   * @return True if zero-length array
   */
  bool is_zero_length_array(const exprt &expr);

  /**
   * @brief Get base address of array (first element)
   * @param arr Array expression
   * @return Address expression
   */
  exprt get_array_base_address(const exprt &arr);

  /**
   * Convert a string to an integer using Python's int() function
   * @param string_obj The string expression to convert
   * @param base_arg The base for conversion (2-36, or 0 for auto-detect)
   * @param location Source location for error reporting
   * @return Expression representing the integer result
   */
  exprt handle_string_to_int(
    const exprt &string_obj,
    const exprt &base_arg,
    const locationt &location);

  /**
   * Convert a string to an integer using base 10 (convenience method)
   * @param string_obj The string expression to convert
   * @param location Source location for error reporting
   * @return Expression representing the integer result
   */
  exprt handle_string_to_int_base10(
    const exprt &string_obj,
    const locationt &location);

  /**
   * Handle int() builtin with automatic type detection
   * Supports int(x) where x can be: int, float, bool, or string
   * @param arg The argument to convert
   * @param location Source location for error reporting
   * @return Expression representing the integer result
   */
  exprt handle_int_conversion(const exprt &arg, const locationt &location);

  /**
   * Handle int() builtin with explicit base
   * Supports int(x, base) where x is a string
   * @param arg The string argument to convert
   * @param base The base for conversion
   * @param location Source location for error reporting
   * @return Expression representing the integer result
   */
  exprt handle_int_conversion_with_base(
    const exprt &arg,
    const exprt &base,
    const locationt &location);

  /**
   * Handle chr() builtin function
   * Converts a Unicode code point to its string representation
   * Supports chr(i) where i is an integer in range [0, 0x10FFFF]
   * @param codepoint_arg The integer code point to convert
   * @param location Source location for error reporting
   * @return Expression representing the resulting string (char array pointer)
   */
  exprt
  handle_chr_conversion(const exprt &codepoint_arg, const locationt &location);

private:
  python_converter &converter_;
  contextt &symbol_table_;
  type_handler &type_handler_;
  string_builder *string_builder_;

  // Helper methods for internal use

  /**
   * @brief Create a character array expression
   * @param chars Character data
   * @param type Array type
   * @return Character array expression
   */
  exprt make_char_array_expr(
    const std::vector<unsigned char> &chars,
    const typet &type);

  /**
   * @brief Find or create a function symbol for string operations
   * @param function_name Name of the function
   * @param return_type Return type of function
   * @param arg_types Argument types
   * @param location Source location
   * @return Function identifier
   */
  std::string ensure_string_function_symbol(
    const std::string &function_name,
    const typet &return_type,
    const std::vector<typet> &arg_types,
    const locationt &location);

  /**
   * @brief Convert float bits to string representation
   * @param float_bits Binary representation of float
   * @param width Bit width (32 or 64)
   * @param precision Number of decimal places
   * @return String representation
   */
  std::string float_to_string(
    const std::string &float_bits,
    std::size_t width,
    int precision);
};

#endif // PYTHON_FRONTEND_STRING_HANDLER_H