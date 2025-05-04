#pragma once

#include <python-frontend/python_converter.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/symbol_id.h>
#include <util/expr.h>
#include <nlohmann/json.hpp>

enum class FunctionType
{
  Constructor,
  ClassMethod,
  InstanceMethod,
  FreeFunction,
};

class symbol_id;

class function_call_expr
{
public:
  function_call_expr(
    const symbol_id &function_id,
    const nlohmann::json &call,
    python_converter &converter);

  virtual ~function_call_expr() = default;

  /*
   * Converts the function from the AST into an exprt.
   */
  virtual exprt get();

  const symbol_id &get_function_id() const
  {
    return function_id_;
  }

private:
  /**
   * Determines whether a non-deterministic function is being invoked.
   */
  bool is_nondet_call() const;

  /*
   * Creates an expression for a non-deterministic function call.
   */
  exprt build_nondet_call() const;

  /*
   * Creates a constant expression from function argument.
   */
  exprt build_constant_from_arg() const;

  /*
   * Sets the function_type_ attribute based on the call information.
   */
  void get_function_type();

  /*
   * Retrieves the object (caller) name from the AST.
   */
  std::string get_object_name() const;

  /*
   * Handles string arguments (e.g., str("abc")) by converting them
   * into character array expressions.
   */
  size_t handle_str(nlohmann::json &arg) const;

  /*
   * Handles float-to-int conversions (e.g., int(3.14)) by generating
   * the appropriate cast expression.
   */
  void handle_float_to_int(nlohmann::json &arg) const;

  /*
   * Handles int-to-float conversions (e.g., float(3)) by generating
   * the appropriate cast expression.
   */
  void handle_int_to_float(nlohmann::json &arg) const;

  /*
   * Handles chr(int) conversions by creating a single-character
   * string expression from an integer.
   */
  void handle_chr(nlohmann::json &arg) const;

  /*
   * Handles hexadecimal string arguments (e.g., hex(255) -> "0xff")
   * by building a constant expression representing the string.
   */
  exprt handle_hex(nlohmann::json &arg) const;

protected:
  symbol_id function_id_;
  const nlohmann::json &call_;
  python_converter &converter_;
  const type_handler &type_handler_;
  FunctionType function_type_;
};
