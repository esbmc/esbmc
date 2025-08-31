#pragma once

#include <util/expr.h>
#include <util/type.h>

#include <map>
#include <regex>
#include <string>

enum class StatementType
{
  VARIABLE_ASSIGN,
  COMPOUND_ASSIGN,
  FUNC_DEFINITION,
  IF_STATEMENT,
  WHILE_STATEMENT,
  FOR_STATEMENT,
  EXPR,
  RETURN,
  ASSERT,
  CLASS_DEFINITION,
  PASS,
  IMPORT,
  BREAK,
  CONTINUE,
  RAISE,
  UNKNOWN,
  GLOBAL,
  TRY,
  EXCEPTHANDLER
};

enum class ExpressionType
{
  BINARY_OPERATION,
  UNARY_OPERATION,
  FUNC_CALL,
  IF_EXPR,
  LOGICAL_OPERATION,
  LITERAL,
  SUBSCRIPT,
  VARIABLE_REF,
  LIST,
  UNKNOWN
};

class type_utils
{
public:
  static bool is_builtin_type(const std::string &name)
  {
    return (
      name == "int" || name == "float" || name == "bool" || name == "str" ||
      name == "chr" || name == "hex" || name == "oct" || name == "ord" ||
      name == "abs");
  }

  static bool is_consensus_type(const std::string &name)
  {
    return (
      name == "uint64" || name == "uint256" || name == "Epoch" ||
      name == "Gwei" || name == "BLSFieldElement" || name == "Slot" ||
      name == "GeneralizedIndex");
  }

  static bool is_consensus_func(const std::string &name)
  {
    return consensus_func_to_type().find(name) !=
           consensus_func_to_type().end();
  }

  static std::string get_type_from_consensus_func(const std::string &name)
  {
    if (!is_consensus_func(name))
      return std::string();
    return consensus_func_to_type().at(name);
  }

  static bool is_python_model_func(const std::string &name)
  {
    return (
      name == "ESBMC_range_next_" || name == "ESBMC_range_has_next_" ||
      name == "bit_length" || name == "from_bytes" || name == "to_bytes" ||
      name == "randint" || name == "random");
  }

  static bool is_python_exceptions(const std::string &name)
  {
    return (
      name == "BaseException" || name == "ValueError" || name == "TypeError" ||
      name == "IndexError" || name == "KeyError" ||
      name == "ZeroDivisionError");
  }

  static bool is_c_model_func(const std::string &func_name)
  {
    return func_name == "ceil" || func_name == "floor" || func_name == "fabs" ||
           func_name == "sin" || func_name == "cos" || func_name == "exp" ||
           func_name == "fmod" || func_name == "sqrt" || func_name == "fmin" ||
           func_name == "fmax" || func_name == "trunc" ||
           func_name == "round" || func_name == "copysign" ||
           func_name == "arctan" || func_name == "arccos" ||
           func_name == "dot" || func_name == "add" ||
           func_name == "subtract" || func_name == "multiply" ||
           func_name == "divide" || func_name == "transpose" ||
           func_name == "det" || func_name == "matmul" || func_name == "pow" ||
           func_name == "log" || func_name == "pow_by_squaring" ||
           func_name == "log2" || func_name == "log1p_taylor" ||
           func_name == "ldexp";
  }

  static bool is_ordered_comparison(const std::string &op)
  {
    return op == "Lt" || op == "Gt" || op == "LtE" || op == "GtE";
  }

  static bool is_relational_op(const std::string &op)
  {
    return (
      op == "Eq" || op == "Lt" || op == "LtE" || op == "NotEq" || op == "Gt" ||
      op == "GtE" || op == "And" || op == "Or");
  }

  static bool is_char_type(const typet &t)
  {
    return (t.is_signedbv() || t.is_unsignedbv()) &&
           t.get("#cpp_type") == "char";
  }

  static bool is_float_vs_char(const exprt &a, const exprt &b)
  {
    const auto &type_a = a.type();
    const auto &type_b = b.type();
    return (type_a.is_floatbv() && is_char_type(type_b)) ||
           (type_b.is_floatbv() && is_char_type(type_a));
  }

  // Helper function to get numeric width from type
  static size_t get_type_width(const typet &type)
  {
    // First try to parse width directly
    try
    {
      return std::stoi(type.width().c_str());
    }
    catch (const std::exception &)
    {
      // If direct parsing fails, try to infer from type name
      std::string type_str = type.width().as_string();

      // Handle common Python/ESBMC type mappings
      if (type_str == "int" || type_str == "int32")
        return 32;
      else if (type_str == "int64" || type_str == "long")
        return 64;
      else if (type_str == "int16" || type_str == "short")
        return 16;
      else if (type_str == "int8" || type_str == "char")
        return 8;
      else if (type_str == "float" || type_str == "float32")
        return 32;
      else if (type_str == "double" || type_str == "float64")
        return 64;
      else if (type_str == "bool")
        return 1;

      // Try to extract number from string like "int32", "uint64", etc.
      std::regex width_regex(R"(\d+)");
      std::smatch match;
      if (std::regex_search(type_str, match, width_regex))
      {
        try
        {
          return std::stoi(match.str());
        }
        catch (const std::exception &)
        {
          // Fall through to default
        }
      }

      // Default to 32 for unknown types
      return 32;
    }
  }

private:
  static const std::map<std::string, std::string> &consensus_func_to_type()
  {
    static const std::map<std::string, std::string> func_to_type = {
      {"hash", "uint256"}};
    return func_to_type;
  }
};
