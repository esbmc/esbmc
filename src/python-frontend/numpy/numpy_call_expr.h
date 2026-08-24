#pragma once

#include <python-frontend/function_call/expr.h>
#include <nlohmann/json.hpp>
#include <optional>

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

  // Looks up a keyword argument by name in the call's "keywords" array (e.g.
  // offset=/axis1=/dtype=), or nullptr if absent. Shared by every 2-D-only
  // view/reduction dispatch (diagonal/trace/fill_diagonal) that rejects
  // out-of-scope keyword arguments.
  const nlohmann::json *find_keyword_arg(const std::string &name) const;

  // Routes to the below for "diagonal"/"trace"/"fill_diagonal"/"ravel",
  // std::nullopt otherwise. Split out of get() for the same reason as
  // get_arange_expr(): keeping that function's own decision count from
  // growing further as ADR-NP-003 etapa 2 lands more pointer-view call
  // forms.
  std::optional<exprt> try_get_pointer_view_call_result();

  // Literal offset= for diagonal/trace, shared since both accept it as
  // either the 2nd positional argument or the offset= keyword and reject
  // anything non-constant the same way. error_context names the caller
  // ("diagonal"/"trace") in the thrown TypeError.
  long long extract_literal_diagonal_offset(const char *error_context);

  exprt handle_diagonal_call();
  exprt handle_trace_call();
  exprt handle_fill_diagonal_call();
  std::optional<exprt> handle_ravel_pointer_view_attempt();
};
