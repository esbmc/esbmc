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

  exprt build();

private:
  void build_function_id();

  bool is_nondet_call() const;

  bool is_assume_call() const;

  bool is_len_call() const;

  exprt build_nondet_call() const;

  exprt build_constant_from_arg() const;

  void get_function_type();

  std::string get_object_name() const;

  const nlohmann::json &call_;
  python_converter &converter_;
  const type_handler &type_handler_;
  symbol_id function_id_;
  FunctionType function_type_;
};
