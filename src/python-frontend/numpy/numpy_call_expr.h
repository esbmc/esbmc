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

  // The construction site's own entry point (function_call_builder::build):
  // applies try_hoist_call_arg_for_view_method ahead of get() the same way
  // get() itself used to, but from outside the class so the check's own
  // decision point is attributed to this small new function instead of
  // inflating get()'s already far-over-threshold decision count (see
  // numpy_call_expr.cpp for the full rationale).
  static exprt build_result(
    const symbol_id &function_id,
    const nlohmann::json &call,
    python_converter &converter);

private:
  exprt create_expr_from_call();

  // transpose()/flatten()/ravel() and every other descriptor-materialized
  // dispatch (sum/mean/min/max/argsort/searchsorted) over a raw Call
  // argument (e.g. `np.eye(3).transpose()`, rewritten to
  // `np.transpose(np.eye(3))`): hoists it into a temp so the rest of get()
  // sees the already-correct Name case. nullopt (not this shape, or
  // hoisting declined) when get()'s normal dispatch should run unchanged.
  // See numpy_call_expr.cpp for the full rationale, including which methods
  // had to stay excluded (their own Name-argument resolution walks the
  // source AST rather than the descriptor map, so it can't see a temp that
  // exists only in the GOTO IR).
  std::optional<exprt>
  try_hoist_call_arg_for_view_method(const std::string &function);

  // A dispatch whose result shape the static annotator cannot model (e.g.
  // ravel()/flatten(), always 1-D of size = product of the input's dims)
  // reaches an assignment target still carrying the annotator's guess --
  // typically Any/void* -- instead of the concrete type just computed.
  // Retyping the target in place here, the same fixup transpose's own
  // dispatch applies inline (try_transpose_name_arg), is what makes an
  // assignment store the array value directly instead of decaying it to a
  // pointer to match the stale declared type; every subsequent subscript on
  // it would otherwise resolve as a symbolic NONDET rather than the actual
  // element. A separate function (not inlined into get()'s own flatten/
  // ravel branch) so this decision point is attributed here instead of
  // adding to get()'s already far-over-threshold count.
  static exprt
  retype_current_lhs_and_return(python_converter &converter, exprt value);

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

  // `*_like`'s element type from dtype= (an override) or the base array's
  // own type. Split out of get() to keep that function's own decision count
  // from growing further.
  typet resolve_like_element_type(const typet &base_type);

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

  // numpy.transpose()'s parameter-shaped fast path: when `t` (the single-
  // pointer-unwrapped type of `arg`) isn't a fully nested 2-D array -- most
  // commonly a 2-D parameter, whose C-ABI row-pointer decay
  // (register_function_argument) loses the outer dimension -- rebuilds a
  // genuine nested array from the parameter's tracked full shape
  // (numpy_param_shapes_) and returns the transposed value directly.
  // nullopt for anything the descriptor materialization declines (rank 1,
  // non-2-D, or not a tracked array at all), leaving the caller's own
  // fully-nested-array handling (e.g. a local array) unchanged. Split out of
  // create_expr_from_call to keep that function's own decision count down.
  std::optional<exprt>
  try_transpose_decayed_2d_param(const nlohmann::json &arg, typet t);

  // The full body of create_expr_from_call's `function == "transpose"`
  // dispatch over a resolved Name argument: the parameter-shaped fast path
  // above, the fully-nested 2-D case (materializing via a C-call or a
  // direct value depending on whether current_lhs exists yet), and the
  // already-1-D passthrough. nullopt when none apply, so the caller falls
  // through to the shared list_arg handling below unchanged. Split out of
  // create_expr_from_call to keep that function's own decision count down.
  std::optional<exprt>
  try_transpose_name_arg(const nlohmann::json &arg, const exprt &arg_expr);

  // np.argsort()/np.searchsorted()/np.sort()'s full dispatch bodies (axis/
  // keyword parsing, the axis-aware/view-aware descriptor path, and the
  // literal-only fallback). Each always returns or throws, never falls
  // through, so get() calling it is a one-for-one replacement of its own
  // former `if (function == "...")` body -- moving that body's decision
  // count out of get() to keep that function's own count from growing.
  exprt handle_argsort_call();
  exprt handle_searchsorted_call();
  exprt handle_sort_call();

  // numpy.argsort()'s axis argument: positional (2nd arg) or axis= keyword,
  // never both. Split out of handle_argsort_call to keep that function's
  // own decision count down.
  const nlohmann::json *resolve_argsort_axis_node() const;

  // numpy.sort()'s axis argument: positional (2nd arg) or axis= keyword,
  // never both; None flattens, otherwise a literal integer, or throws.
  // Split out of handle_sort_call to keep that function's own decision
  // count down.
  void parse_sort_axis_and_keywords(bool &flatten, long long &axis);
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

  // Resolves `raw_arg` -- a Name already bound to a concrete numpy array, or
  // a call to a user function returning one (direct, or via a local
  // variable) -- through the same descriptor-materialization path
  // sort()/argsort() already use for a Name, instead of searchsorted's own
  // AST-literal-tracing-only resolution. See numpy_call_expr.cpp for the
  // full rationale.
  std::optional<std::vector<exprt>>
  resolve_searchsorted_array_via_descriptor(const nlohmann::json &raw_arg);

  // sorter=argsort(<the same array>) over a descriptor-resolved array: an
  // exprt-level stable-sort gather (bubble_sort_numpy_paired) over
  // resolve_searchsorted_array_via_descriptor's elements, since they are
  // index expressions into a local array and almost never compile-time
  // constant the way a genuine AST literal's elements would be. nullopt for
  // anything else (a different array's argsort, a literal index array, an
  // array the descriptor path itself can't resolve). See numpy_call_expr.cpp
  // for the full rationale.
  std::optional<std::vector<exprt>>
  resolve_searchsorted_sorted_values_via_descriptor(
    const nlohmann::json &raw_arg,
    const nlohmann::json &sorter_node,
    const std::string &array_name);

  // Evaluates `call_node` (a call to a user function) exactly once by
  // synthesizing `<temp> = call_node` and converting it through the normal
  // assignment pipeline, so side effects execute once and the temp's numpy
  // array metadata is registered like any other local-array-return
  // assignment. Returns a Name node referencing the temp; nullopt when
  // there is no current block to emit into.
  std::optional<nlohmann::json>
  hoist_call_argument_into_temp(const nlohmann::json &call_node);

  // handle_searchsorted_call's dispatch once its array argument resolved via
  // resolve_searchsorted_array_via_descriptor (no sorter= given -- see that
  // function). Split out to keep handle_searchsorted_call's own decision
  // count down.
  exprt handle_searchsorted_call_over_descriptor(
    std::vector<exprt> values,
    bool right);

  // A same-shaped placeholder for handle_searchsorted_call's array argument
  // when it is a user function call reached during a discarded type-probe
  // pass (hoisting would evaluate it an extra time); nullopt otherwise. See
  // numpy_call_expr.cpp for the full rationale. Split out to keep
  // handle_searchsorted_call's own decision count down.
  std::optional<exprt> try_searchsorted_probe_placeholder();

  // The AST-literal resolution only, declining (nullopt) rather than
  // throwing. See numpy_call_expr.cpp for the full rationale. Split out to
  // keep handle_searchsorted_call's own decision count down.
  std::optional<nlohmann::json>
  try_resolve_searchsorted_literal_array(const std::string &function);

  // resolve_searchsorted_array_via_descriptor plus the dispatch to
  // handle_searchsorted_call_over_descriptor, as a single nullopt-on-decline
  // step. Split out to keep handle_searchsorted_call's own decision count
  // down.
  std::optional<exprt> try_searchsorted_call_over_descriptor(bool right);

  // Validates `arr_arg`'s shape and applies `sorter_node`/sortedness,
  // returning the space handle_searchsorted_call_over_literal searches. See
  // numpy_call_expr.cpp for the full rationale. Split out to keep
  // handle_searchsorted_call's own decision count down.
  nlohmann::json resolve_searchsorted_space(
    nlohmann::json arr_arg,
    const nlohmann::json *sorter_node,
    const std::string &array_name);

  // handle_searchsorted_call's final step over an AST-literal
  // `search_space`. Split out to keep handle_searchsorted_call's own
  // decision count down.
  exprt handle_searchsorted_call_over_literal(
    nlohmann::json search_space,
    bool right);

  // np.sum(identity(x))/np.argmin(identity(x)): a reducer's argument reaches
  // get() as a raw Call node when it is itself a nested call, never through
  // function_call_expr's own dispatch (where try_fold_identity_array_return
  // handles the equivalent `y = identity(x)` case). Reuses the same
  // converter-level substitution for a pure, argument-only return (e.g.
  // `def identity(a): return a`), inlining it to that substituted
  // expression -- typically a bare Name the caller's own resolve_var can
  // then resolve as usual -- so it is left unchanged for anything else
  // (multi-statement bodies, non-argument locals, keyword calls).
  nlohmann::json try_inline_pure_call_arg(nlohmann::json arg) const;
};
