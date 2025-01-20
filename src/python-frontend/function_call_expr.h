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

class function_call_expr
{
public:
  function_call_expr(const nlohmann::json &call, python_converter &converter);

  /*
   * Converts the function from the AST into an exprt.
   */
  exprt build();

private:
  /*
   * Extracts information from the call to populate the function_id attribute.
   */
  void build_function_id();

  /**
   * Determines whether a non-deterministic function is being invoked.
   */
  bool is_nondet_call() const;

  /*
   * Checks if assume (__ESBMC_assume or __VERIFIER_assume) function is being invoked.
   */
  bool is_assume_call() const;

  /*
   * Checks if the Python len() function is being invoked.
   */
  bool is_len_call() const;

  /*
   * Checks if a NumPy function is being invoked.
   */
  bool is_numpy_call() const;

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

  const nlohmann::json &call_;
  python_converter &converter_;
  const type_handler &type_handler_;
  symbol_id function_id_;
  FunctionType function_type_;
};
