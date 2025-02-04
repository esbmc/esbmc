#pragma once

#include <nlohmann/json.hpp>

class symbol_id;
class exprt;
class python_converter;

class numpy_call_expr
{
public:
  numpy_call_expr(
    const symbol_id &function_id,
    const nlohmann::json &call,
    python_converter &converter);

  exprt get();

private:
  const symbol_id &function_id_;
  const nlohmann::json &call_;
  python_converter &converter_;
};
