#pragma once

#include <python-frontend/function_call/expr.h>
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

  ~numpy_call_expr();

  exprt get() override;

private:
  exprt create_expr_from_call();

  // np.arange(...) with constant, small arguments materialized to a literal
  // list, falling back to the operational model for genuinely non-constant
  // arguments (see numpy_call_expr.cpp for the full rationale). Split out of
  // get() to keep that function's own decision count from growing further.
  exprt get_arange_expr();

  bool is_math_function() const;

  void broadcast_check(const nlohmann::json &operands) const;

  std::string get_dtype() const;
  typet get_typet_from_dtype() const;
  size_t get_dtype_size() const;
};
