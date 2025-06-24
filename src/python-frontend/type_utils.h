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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
      name == "chr" || name == "hex" || name == "oct" || name == "ord" ||
      name == "abs");
=======
      name == "chr");
>>>>>>> 1078ace71 ([Python]  add chr() built-in function support (#2416))
=======
      name == "chr" || name == "hex");
>>>>>>> 8a6c3ff59 ([python] Add support for hex() built-in function (#2419))
=======
      name == "chr" || name == "hex" || name == "oct");
>>>>>>> 038fae2a5 ([python] add support for python's oct() built-in function (#2421))
=======
      name == "chr" || name == "hex" || name == "oct" || name == "ord");
>>>>>>> ffcecbd12 ([python] Handle ord(): convert single-character string to integer Unicode code point (#2423))
=======
      name == "chr" || name == "hex" || name == "oct" || name == "ord" ||
      name == "abs");
>>>>>>> e73ba5595 ([python]  add support and error checking for Python abs() builtin function (#2441))
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 5cd8a6f44 ([python] Allow variables in NumPy math function calls (#2469))
    return func_name == "ceil" || func_name == "floor" || func_name == "fabs" ||
           func_name == "sin" || func_name == "cos" || func_name == "exp" ||
           func_name == "fmod" || func_name == "sqrt" || func_name == "fmin" ||
           func_name == "fmax" || func_name == "trunc" ||
           func_name == "round" || func_name == "copysign" ||
           func_name == "arctan" || func_name == "arccos" ||
           func_name == "dot" || func_name == "add" ||
           func_name == "subtract" || func_name == "multiply" ||
<<<<<<< HEAD
<<<<<<< HEAD
           func_name == "divide" || func_name == "transpose" ||
           func_name == "det" || func_name == "matmul" || func_name == "pow" ||
           func_name == "log" || func_name == "pow_by_squaring" ||
           func_name == "log2" || func_name == "log1p_taylor" ||
           func_name == "ldexp";
<<<<<<< HEAD
=======
    return (func_name == "ceil");
>>>>>>> f59fd87f9 ([python-frontend] Reuse C libm models for NumPy math functions (#2395))
=======
    return (func_name == "ceil") || (func_name == "floor") ||
           (func_name == "fabs") || (func_name == "sin") ||
           (func_name == "cos") || (func_name == "exp") ||
<<<<<<< HEAD
           (func_name == "fmod");
>>>>>>> 2e5aef15f ([python] Expand numpy math models (#2407))
=======
           (func_name == "fmod") || (func_name == "sqrt") ||
           (func_name == "fmin") || (func_name == "fmax") ||
           (func_name == "trunc") || (func_name == "round") ||
<<<<<<< HEAD
           (func_name == "copysign" || func_name == "arctan");
>>>>>>> f86e3ad14 ([python] added test cases for numpy math functions (#2437))
=======
           (func_name == "copysign" || func_name == "arctan") ||
<<<<<<< HEAD
           (func_name == "arccos");
>>>>>>> 227449899 ([numpy] Add support for numpy.arccos() and some fixes (#2444))
=======
           (func_name == "arccos") || (func_name == "dot");
>>>>>>> bea8feb3b ([python] Handling NumPy dot product (#2460))
=======
           func_name == "divide";
>>>>>>> 5cd8a6f44 ([python] Allow variables in NumPy math function calls (#2469))
=======
           func_name == "divide" || func_name == "transpose" ||
<<<<<<< HEAD
           func_name == "det";
>>>>>>> 26666f4c8 ([python] Handling numpy functions (#2474))
=======
           func_name == "det" || func_name == "matmul";
>>>>>>> 80338d5a3 ([python] Add numpy.matmul (#2487))
=======
>>>>>>> 59fb1bf8f ([python] enhanced handling of true division (#2505))
  }

private:
  static const std::map<std::string, std::string> &consensus_func_to_type()
  {
    static const std::map<std::string, std::string> func_to_type = {
      {"hash", "uint256"}};
    return func_to_type;
  }
};
