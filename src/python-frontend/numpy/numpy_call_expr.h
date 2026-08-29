#pragma once

#include <python-frontend/function_call/expr.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <utility>
#include <vector>

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
  exprt handle_axis_permutation_view_call(const std::string &function);
  exprt handle_broadcast_to_call();
  std::optional<exprt>
  try_build_nditer_descriptor_list(const nlohmann::json &arg);
  std::optional<exprt> try_materialize_descriptor_copy_call();
  std::optional<exprt>
  try_materialize_descriptor_array_call(nlohmann::json &array_arg);
  std::optional<exprt> try_reduce_descriptor_call(const std::string &function);
  std::optional<exprt> try_reduce_descriptor_call_along_axis(
    const std::string &function,
    const std::pair<std::vector<std::size_t>, std::vector<exprt>>
      &materialized);
  void reject_unsupported_nditer_keywords(const nlohmann::json &arg) const;
  void reject_unsupported_transpose_axes_rank(const std::string &function);

  // sum/prod/min/max/mean/argmin/argmax's flattened fallback path (a
  // genuine inline literal, or argmin/argmax, which never go through
  // try_reduce_descriptor_call's own keyword check). Split out of get() for
  // the same reason as get_arange_expr()/try_get_pointer_view_call_result().
  void reject_unsupported_flattened_reducer_keywords(
    const std::string &function) const;

  // argmin/argmax's own axis= handling: they have no descriptor-call fast
  // path (try_reduce_descriptor_call only covers sum/mean/min/max), so this
  // is checked directly against the already-resolved array node, ahead of
  // reject_unsupported_flattened_reducer_keywords's generic "no keywords at
  // all" rejection. Returns nullopt (no axis given) to fall through to that
  // existing flattened path unchanged.
  std::optional<exprt> try_argmin_argmax_along_axis(
    const std::string &function,
    const nlohmann::json &arg);

  // One-line dispatch guard so adding argmin/argmax's axis fast path does not
  // grow get()'s own decision count -- same reasoning as get_arange_expr()/
  // try_get_pointer_view_call_result().
  std::optional<exprt> try_argmin_argmax_axis_result(
    const std::string &function,
    const nlohmann::json &arg);

  // sum/prod have a defined identity result over zero elements; every other
  // flattened reducer must reject instead. Split out of get() to keep that
  // function's own decision count from growing further.
  exprt empty_reducer_identity_result(const std::string &function) const;

  // np.any(a, ...)/np.all(a, ...) are dispatched here (is_numpy_call()
  // routes any np.<attr>(...) call through numpy_call_expr::get() before
  // function_call_expr's own table-driven dispatch -- the one that would
  // otherwise reach handle_any()/handle_all() -- ever runs). One-line
  // dispatch guard for the same reason as get_arange_expr()/
  // try_get_pointer_view_call_result().
  std::optional<exprt> try_any_all_result(const std::string &function);

  // Resolves median/percentile/argsort/searchsorted's array argument to a
  // literal List node: a List stays as-is; a Name is followed to its own
  // declaration's value only when inline_only is false AND the name has a
  // single assignment in scope (find_var_decl() otherwise returns the first
  // textual assignment, not the one reaching this call -- see
  // argsort_reassigned_array_fail). Throws when the (possibly resolved)
  // argument still isn't a literal List. A member function rather than a
  // lambda local to get() so its own decision count is attributed here
  // instead of inflating get()'s.
  nlohmann::json resolve_literal_numpy_array_input(
    nlohmann::json arr_arg,
    const std::string &function_name,
    bool inline_only = false);
};
