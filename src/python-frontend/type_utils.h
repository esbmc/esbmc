#pragma once

#include <map>
#include <string>

enum class StatementType
{
  VARIABLE_ASSIGN,
  COMPOUND_ASSIGN,
  FUNC_DEFINITION,
  IF_STATEMENT,
  WHILE_STATEMENT,
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

private:
  static const std::map<std::string, std::string> &consensus_func_to_type()
  {
    static const std::map<std::string, std::string> func_to_type = {
      {"hash", "uint256"}};
    return func_to_type;
  }
};
