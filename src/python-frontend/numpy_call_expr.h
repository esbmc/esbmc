#pragma once

#include <python-frontend/function_call_expr.h>
#include <nlohmann/json.hpp>

class symbol_id;
class exprt;
class typet;
class python_converter;

class numpy_call_expr : public function_call_expr
{
public:
  numpy_call_expr(
    const symbol_id &function_id,
    const nlohmann::json &call,
    python_converter &converter);

  exprt get() override;

private:
  exprt create_expr_from_call();

  bool is_math_function() const;

  void broadcast_check(const nlohmann::json &operands) const;

  std::string get_dtype() const;
  typet get_typet_from_dtype() const;
  size_t get_dtype_size() const;
};
