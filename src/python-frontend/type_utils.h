#pragma once

#include <util/c_types.h>
#include <util/expr.h>
#include <util/expr_util.h>
#include <util/type.h>

#include <nlohmann/json.hpp>

#include <map>
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
  EXCEPTHANDLER,
  DELETE
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
  UNKNOWN,
  FSTRING,
  TUPLE
};

struct TypeFlags
{
  bool has_float = false;
  bool has_int = false;
  bool has_bool = false;
};

class type_utils
{
public:
  static bool is_builtin_type(const std::string &name)
  {
    return (
      name == "int" || name == "float" || name == "bool" || name == "str" ||
      name == "chr" || name == "hex" || name == "oct" || name == "ord" ||
      name == "abs" || name == "tuple" || name == "list" || name == "dict" ||
      name == "set" || name == "frozenset" || name == "bytes" ||
      name == "set" || name == "bytearray" || name == "range" ||
      name == "complex" || name == "type" || name == "object" ||
      name == "None" || name == "divmod");
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
      name == "BaseException" || name == "Exception" || name == "ValueError" ||
      name == "TypeError" || name == "IndexError" || name == "KeyError" ||
      name == "ZeroDivisionError" || name == "AssertionError" ||
      name == "NameError" || name == "OSError" || name == "FileNotFoundError" ||
      name == "FileExistsError" || name == "PermissionError" ||
      name == "NotImplementedError");
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

  static std::string remove_quotes(const std::string &str)
  {
    if (str.length() < 2)
      return str;

    // Check for single quotes
    if (str.front() == '\'' && str.back() == '\'')
      return str.substr(1, str.length() - 2);

    // Check for double quotes
    if (str.front() == '"' && str.back() == '"')
      return str.substr(1, str.length() - 2);

    // No quotes found, return original string
    return str;
  }

  // Select widest type from flags based on hierarchy: float > int > bool
  static typet
  select_widest_type(const TypeFlags &flags, const typet &default_type)
  {
    if (flags.has_float)
      return double_type();
    if (flags.has_int)
      return long_long_int_type();
    if (flags.has_bool)
      return bool_type();

    return default_type;
  }

  // Extract type flags from Union annotation slice
  static TypeFlags extract_union_types(const nlohmann::json &slice)
  {
    TypeFlags flags;

    if (slice["_type"] == "Tuple" && slice.contains("elts"))
    {
      for (const auto &elem : slice["elts"])
        update_type_flags_from_node(elem, flags);
    }

    return flags;
  }

  // Extract type flags from PEP 604 union syntax: int | bool
  static TypeFlags extract_binop_union_types(const nlohmann::json &binop_node)
  {
    TypeFlags flags;

    // Extract from left operand
    if (binop_node.contains("left"))
      update_type_flags_from_node(binop_node["left"], flags);

    // Extract from right operand (may be nested BinOp for chained unions)
    if (binop_node.contains("right"))
    {
      const auto &right = binop_node["right"];
      if (right["_type"] == "BinOp")
      {
        // Recursively handle chained unions: int | bool | float
        TypeFlags right_flags = extract_binop_union_types(right);
        merge_type_flags(flags, right_flags);
      }
      else
        update_type_flags_from_node(right, flags);
    }

    return flags;
  }

  static bool is_integer_type(const typet &type)
  {
    return type.is_signedbv() || type.is_unsignedbv();
  }

  static bool is_string_type(const typet &type)
  {
    // String types are represented as arrays or pointers to char
    if (type.is_array() && type.subtype() == char_type())
      return true;

    if (type.is_pointer())
    {
      const typet &subtype = type.subtype();
      // Direct pointer to char: char*
      if (subtype == char_type())
        return true;
      // Pointer to char array: char(*)[N] (string literals)
      if (subtype.is_array() && subtype.subtype() == char_type())
        return true;
    }

    return false;
  }

private:
  static void
  update_type_flags_from_node(const nlohmann::json &node, TypeFlags &flags)
  {
    if (node["_type"] == "Name" && node.contains("id"))
    {
      const std::string type_str = node["id"].get<std::string>();
      if (type_str == "float")
        flags.has_float = true;
      else if (type_str == "int")
        flags.has_int = true;
      else if (type_str == "bool")
        flags.has_bool = true;
    }
  }

  static void merge_type_flags(TypeFlags &dest, const TypeFlags &src)
  {
    dest.has_float = dest.has_float || src.has_float;
    dest.has_int = dest.has_int || src.has_int;
    dest.has_bool = dest.has_bool || src.has_bool;
  }

  static const std::map<std::string, std::string> &consensus_func_to_type()
  {
    static const std::map<std::string, std::string> func_to_type = {
      {"hash", "uint256"}};
    return func_to_type;
  }
};
