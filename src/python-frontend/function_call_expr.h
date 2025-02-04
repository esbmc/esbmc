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

  /*
   * Converts the function from the AST into an exprt.
   */
  exprt get();

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

  symbol_id function_id_;
  const nlohmann::json &call_;
  python_converter &converter_;
  const type_handler &type_handler_;
  FunctionType function_type_;
};
