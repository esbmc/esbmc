#pragma once

#include <nlohmann/json.hpp>
#include <python-frontend/type/element_type_registry.h>
#include <util/irep/type.h>
#include <util/irep/expr.h>
#include <util/irep/std_expr.h>
#include <util/symtab/symbol.h>
#include <functional>
#include <optional>
#include <set>
#include <utility>

class exprt;
class symbolt;
class python_converter;
class type_handler;
class codet;

struct list_elem_info
{
  symbolt *elem_type_sym;
  symbolt *elem_symbol;
  exprt elem_size;
  locationt location;
};

struct flat_array_shape_info
{
  typet elem_type;
  long long total_length;
};

class python_list
{
public:
  python_list(python_converter &converter, const nlohmann::json &list)
    : converter_(converter), list_value_(list)
  {
  }

  // @p from_right selects rsplit() semantics (split at the rightmost @p count
  // separators) over split() (the leftmost). Defaults to split().
  static exprt build_split_list(
    python_converter &converter,
    const nlohmann::json &call_node,
    const std::string &input,
    const std::string &separator,
    long long count,
    bool from_right = false);

  static exprt build_split_list(
    python_converter &converter,
    const nlohmann::json &call_node,
    const exprt &input_expr,
    const std::string &separator,
    long long count,
    bool from_right = false);

  exprt get();

  /**
   * @brief Materialize a fresh list from already-evaluated element values.
   *
   * Unlike get(), which converts AST element nodes, this builds the list from
   * exprt values directly — used to emit the result of a frontend-computed
   * sort over tuples, where each element is a conditional (ite) selection over
   * the input elements. Records each element's type in the type map so later
   * subscripting recovers the element type. The constructor's @c list_value_
   * node supplies the source location.
   */
  exprt build_list_from_exprs(const std::vector<exprt> &elems);

  /**
   * @brief Build a runtime PyListObject filled with @p fill_value repeated
   * @p size times. @p size may be a symbolic expression; the resulting while-
   * loop is bounded by the model checker's --unwind setting. A runtime guard
   * rejects negative sizes with ValueError before the unsigned cast.
   * @param size     Expression giving the number of elements (often symbolic).
   * @param fill_value Element value pushed on each iteration.
   * @param elem_type  IRep2 type of @p fill_value, recorded in the registry.
   * @param index_base When non-null, each element is @p index_base plus the
   *        loop index rather than @p fill_value, which is what a range needs.
   */
  exprt build_symbolic_fill_list(
    const exprt &size,
    const exprt &fill_value,
    const typet &elem_type,
    const exprt *index_base = nullptr);

  exprt index(const exprt &array, const nlohmann::json &slice_node);

  /**
   * @brief Lower boolean-mask indexing `a[mask]` to a runtime loop that
   * builds a fresh list holding the elements of @p array whose matching
   * @p mask entry is True (NumPy fancy-indexing semantics). Both operands
   * must be fixed-size arrays (the numpy array model); a compile-time
   * length mismatch between @p array and @p mask is rejected explicitly.
   * @param array  Source 1-D array expression; multi-dimensional sources are
   *               rejected with TypeError at conversion time.
   * @param mask   Boolean array expression, same length as @p array.
   * @param element The Subscript AST node, used for location info.
   */
  exprt build_bool_mask_index(
    const exprt &array,
    const exprt &mask,
    const nlohmann::json &element);

  /**
   * @brief Lower whole-row boolean-mask selection `a[mask]` on a 2-D array.
   * The runtime-list model used by build_bool_mask_index cannot hold an
   * array-typed element (confirmed empirically: pushing a row produces a
   * bit-vector/array sort mismatch at the SMT backend), so this takes a
   * different path: when @p mask resolves to a concrete boolean literal
   * (`np.array([True, False, ...])`) whose declaring assignment is found via
   * AST lookup, the selected row count is known at conversion time and the
   * result is a fixed-size array, mirroring build_column_select. Otherwise
   * (a symbolic/reassigned mask), it delegates to
   * build_bool_mask_row_select_symbolic.
   * @param array   Source 2-D array expression.
   * @param mask    Boolean mask array expression.
   * @param element The Subscript AST node, used for location info and to
   *                recover the mask's variable name.
   */
  exprt build_bool_mask_row_select(
    const exprt &array,
    const exprt &mask,
    const nlohmann::json &element);

  /**
   * @brief Lower whole-row boolean-mask selection `a[mask]` on a 2-D array
   * for a symbolic (non-literal) mask: the result is the canonical bounded
   * descriptor shape — a worst-case-sized `rows` buffer (capacity ==
   * @p array's row count) plus a runtime `count` member holding the number
   * of rows actually selected, so the logical size is part of the modelled
   * value rather than a detached counter. A single runtime while-loop scans
   * @p mask once, copying each selected row (column by column, since
   * whole-row assignment isn't valid GOTO) into the next free `rows` slot
   * and incrementing `count`, preserving input order.
   * @param array   Source 2-D array expression.
   * @param mask    Boolean mask array expression, same length as @p array's
   *                row count.
   * @param element The Subscript AST node, used for location info.
   */
  exprt build_bool_mask_row_select_symbolic(
    const exprt &array,
    const exprt &mask,
    const nlohmann::json &element);

  /**
   * @brief True if @p type is a result struct built by
   * build_bool_mask_row_select_symbolic (identified by its `tag-` prefix,
   * mirroring tuple_handler::is_tuple_type).
   */
  static bool is_bool_mask_rows_type(const typet &type);

  /**
   * @brief Index `b[i]` into a boolean-mask row-selection result (see
   * build_bool_mask_row_select_symbolic): normalizes a negative @p
   * slice_node index against the struct's runtime `count` member (not the
   * `rows` buffer's physical capacity), bounds-checks it against `count`
   * (raising IndexError out of bounds, mirroring build_list_at_call), and
   * returns the selected row.
   * @param array      The boolean-mask row-selection result expression.
   * @param slice_node AST node for the row index (a plain integer index;
   *                    slicing is not supported).
   * @param element    The Subscript AST node, used for location info.
   */
  exprt index_bool_mask_rows(
    const exprt &array,
    const nlohmann::json &slice_node,
    const nlohmann::json &element);

  /**
   * @brief Lower 2-D column selection `a[:, j]` to a bounded copy over every
   * row of a fixed-shape 2-D array, collecting `row[j]` into a fresh 1-D
   * array. @p array must be a fixed-size array of fixed-size rows (numpy
   * static-shape model); anything else is rejected with TypeError.
   * @param array          Source 2-D array expression.
   * @param col_index_node AST node for the column index (axis 1).
   * @param element        The Subscript AST node, used for location info.
   */
  exprt build_column_select(
    const exprt &array,
    const nlohmann::json &col_index_node,
    const nlohmann::json &element);

  /**
   * @brief Lower a strided column slice `a[:, start:stop:step]` on a
   * fixed-shape 2-D array: the literal slice bounds are resolved at conversion
   * time and each selected column is copied into a fresh 2-D result. Bare
   * negative steps other than `-1` are still rejected because their result
   * width would differ from the old full-reversal path.
   * @param array          Source 2-D array expression.
   * @param col_slice_node AST `Slice` node for the column axis (axis 1);
   *                       its `step`, `lower`, and `upper` must be literal
   *                       integers when present.
   * @param element        The Subscript AST node, used for location info.
   */
  exprt build_strided_column_select(
    const exprt &array,
    const nlohmann::json &col_slice_node,
    const nlohmann::json &element);

  /**
   * @brief np.diagonal(a[, offset=k])/a.diagonal([k]): a read-only strided
   * pointer view into a's own buffer (ADR-NP-003 etapa 2). Public so
   * numpy_call_expr.cpp's Call dispatch (where the offset/axis1/axis2
   * argument parsing lives) can build it directly. Declines (returns
   * std::nullopt) for a non-symbol/non-numpy-tracked source or a source
   * that isn't a fixed-shape 2-D array (including 3-D+). When current_lhs
   * is unset (the type-inference pre-pass that runs before the real `d`
   * symbol exists), returns a same-typed placeholder pointer instead of
   * declining, so that pass can read a type and complete -- the real,
   * current_lhs-correct view is built on the unconditional second pass
   * that follows for a Call RHS.
   * @param array Source 2-D array expression.
   * @param k     Literal diagonal offset (0 = main diagonal).
   */
  std::optional<exprt>
  try_build_diagonal_pointer_view(const exprt &array, long long k);

  /**
   * @brief np.trace(a, offset=k): sum of the elements np.diagonal(a, k)
   * would view, computed directly as a scalar reduction (no view is built).
   * Public for the same reason as try_build_diagonal_pointer_view. Declines
   * for a non-symbol/non-numpy-tracked source or a source that isn't a
   * fixed-shape 2-D array.
   * @param array Source 2-D array expression.
   * @param k     Literal diagonal offset (0 = main diagonal).
   */
  std::optional<exprt> build_trace_reduction(const exprt &array, long long k);

  /**
   * @brief np.fill_diagonal(a, value): mutates a's main diagonal in place
   * via converter_.add_instruction, using the same offset/stride math as
   * try_build_diagonal_pointer_view/build_trace_reduction with k=0. `value`
   * is either a scalar expression or a List literal whose length must equal
   * the diagonal's exactly (throws ValueError otherwise, matching NumPy's
   * own broadcasting rule for a 1-D val). Public for the same reason as
   * try_build_diagonal_pointer_view. Declines (returns false) for a
   * non-symbol/non-numpy-tracked source or a source that isn't a
   * fixed-shape 2-D array.
   * @param array      Source 2-D array expression.
   * @param value_node Raw AST node for the value argument.
   */
  bool try_build_fill_diagonal_mutation(
    const exprt &array,
    const nlohmann::json &value_node);

  /**
   * @brief np.ravel(a)/a.ravel(): a writable, contiguous strided pointer
   * view into a's own buffer (ADR-NP-003 etapa 2), for a fixed-shape 1-D or
   * 2-D array. Public for the same reason as try_build_diagonal_pointer_view.
   * Declines (returns std::nullopt) for a non-symbol/non-numpy-tracked
   * source, a source that isn't fixed-shape 1-D/2-D (including 3-D+), or an
   * unassigned RHS (current_lhs unset) outside the discardable
   * type-inference pre-pass -- same current_lhs/in_rhs_type_probe_ handling
   * as try_build_diagonal_pointer_view, for the same reason.
   * @param array Source 1-D or 2-D array expression.
   */
  std::optional<exprt> try_build_ravel_pointer_view(const exprt &array);

  /// One axis of a strided view, in elements: its extent and stride are
  /// constants unless the matching expression is set (a signed 64-bit
  /// temporary computed at run time).
  struct strided_axis
  {
    long long extent = 0;
    exprt extent_expr = nil_exprt();
    long long stride = 0;
    exprt stride_expr = nil_exprt();

    bool symbolic() const
    {
      return extent_expr.is_not_nil() || stride_expr.is_not_nil();
    }
  };

  /// A numpy array or view as a scalar base pointer plus its axes: the
  /// descriptor every N-D view is built from.
  struct strided_view_desc
  {
    exprt base;
    typet elem_type;
    std::vector<strided_axis> axes;
    /// A view of a read-only view (broadcast_to, diagonal) is read-only.
    bool readonly = false;
  };

  std::optional<strided_view_desc> describe_strided_view(const exprt &array);
  std::optional<exprt> try_build_numpy_param_slice_view(
    const exprt &array,
    const typet &resolved_array_type,
    const nlohmann::json &slice_node);
  bool is_runtime_numpy_slice(
    const exprt &array,
    const typet &elem_type,
    const nlohmann::json &slice_node) const;
  std::optional<exprt> unnamed_nd_view_placeholder(
    const strided_view_desc &view,
    bool as_view,
    bool flat_result,
    const char *unnamed_error) const;
  std::optional<exprt> alias_contiguous_ravel(
    const nlohmann::json &arg,
    const strided_view_desc &view);
  exprt emit_view_copy_buffer(
    const nlohmann::json &arg,
    const strided_view_desc &view,
    const exprt &count);
  std::vector<strided_axis> owned_copy_axes(
    const nlohmann::json &arg,
    const strided_view_desc &view,
    bool flatten,
    const exprt &count);
  void emit_reshape_size_guard(
    const nlohmann::json &arg,
    const exprt &count,
    const std::vector<std::size_t> &new_shape);
  void scale_split_axis_strides(
    const nlohmann::json &arg,
    const strided_axis &source,
    std::vector<strided_axis> &axes);
  void reshape_registered_copy(
    const std::string &lhs_id,
    const std::vector<std::size_t> &new_shape);
  exprt emit_view_accumulator(
    const nlohmann::json &arg,
    const typet &type,
    const exprt &init);
  exprt reduce_view_truth(
    bool is_any,
    const nlohmann::json &arg,
    const strided_view_desc &view);
  exprt reduce_view_extreme(
    const std::string &function,
    const nlohmann::json &arg,
    const strided_view_desc &view,
    const exprt &count);
  bool strided_basic_view_declines(
    const exprt &array,
    const std::vector<nlohmann::json> &idx_nodes,
    bool allow_plain_array) const;
  std::optional<strided_axis> runtime_slice_axis(
    const strided_axis &source,
    const nlohmann::json &node,
    exprt &offset);
  std::optional<strided_axis> slice_strided_axis(
    const strided_axis &source,
    const nlohmann::json &node,
    exprt &offset);
  bool index_strided_axis(
    const strided_axis &source,
    const nlohmann::json &node,
    exprt &offset);
  exprt strided_index_offset(
    const strided_view_desc &src,
    const std::vector<nlohmann::json> &indices);
  std::optional<strided_view_desc> contiguous_view_desc(
    const exprt &array,
    const typet &elem_type,
    const std::vector<std::size_t> &shape) const;
  /// Axes of the pointer view registered under `view_id`.
  std::vector<strided_axis> tracked_view_axes(const std::string &view_id) const;

  /// `base + offset` as a registered view with these axes for the assignment
  /// target (offset is a size_type element count).
  exprt register_strided_view(
    const strided_view_desc &source,
    const exprt &offset,
    const std::vector<strided_axis> &axes,
    bool readonly = false);

  /// Independent array holding the elements a view with these axes reads, in
  /// row-major order. `out_shape` (constant axes only) lays them out in
  /// another shape of the same size.
  exprt copy_strided_view(
    const strided_view_desc &source,
    const exprt &offset,
    const std::vector<strided_axis> &axes,
    const std::vector<std::size_t> *out_shape = nullptr);

  static exprt axis_extent(const strided_axis &axis);
  static exprt axis_stride(const strided_axis &axis);

  /// The descriptor of the named view when its extents or strides are known
  /// only at run time; nullopt for anything else.
  std::optional<strided_view_desc>
  symbolic_view_operand(const nlohmann::json &arg);

  /// Element lvalue of a view at a signed 64-bit multi-index.
  exprt view_element(
    const strided_view_desc &view,
    const std::vector<exprt> &index) const;

  /// `idx = 0; while (idx < extent) { body(idx); idx++; }`, with the body's
  /// statements emitted inside the loop.
  void emit_counted_loop(
    const nlohmann::json &node,
    const exprt &extent,
    const std::function<void(const exprt &)> &body);

  /// Nested loops over every index of a view; `leaf` is called once, inside
  /// the innermost body, with the element lvalue.
  void emit_view_loops(
    const nlohmann::json &node,
    const strided_view_desc &view,
    const std::function<void(const exprt &)> &leaf);

  exprt
  view_element_count(const nlohmann::json &node, const strided_view_desc &view);

  /// sum / mean / min / max / any / all of a run-time-extent view, as loops.
  std::optional<exprt> try_reduce_symbolic_view(
    const std::string &function,
    const nlohmann::json &arg);

  /// A Python list of a run-time-extent view's elements (nested per axis when
  /// `nested`, flat otherwise).
  std::optional<exprt>
  try_build_symbolic_view_list(const nlohmann::json &arg, bool nested);

  enum class view_contiguity
  {
    contiguous,
    not_contiguous,
    unknown // depends on a run-time stride
  };

  /// Whether a view covers its storage densely in row-major order.
  static view_contiguity
  classify_view_contiguity(const strided_view_desc &view);

  /// reshape of a run-time-extent view to constant `new_shape`: an alias when
  /// it is contiguous, a copy otherwise; ValueError if the sizes differ.
  std::optional<exprt> try_reshape_symbolic_view(
    const nlohmann::json &arg,
    const std::vector<std::size_t> &new_shape);

  /// np.copy / np.array / .copy of a run-time-extent view.
  std::optional<exprt> try_copy_symbolic_view(
    const nlohmann::json &arg,
    bool flatten = false,
    bool alias_if_contiguous = false);

  /// Signed 64-bit temporary holding `value`, assigned once here.
  exprt emit_ll_temp(
    const nlohmann::json &node,
    const char *name,
    const exprt &value);

  /// `v = a[i, :, 1:3, ...]` over a rank-3+ array or any N-D view, with
  /// literal or run-time indices and slice bounds: a view into the source's
  /// own storage (aliasing in both directions) when assigned to a bare name,
  /// otherwise an independent copy. Declines for anything else.
  std::optional<exprt> try_build_strided_basic_view(
    const exprt &array,
    const std::vector<nlohmann::json> &idx_nodes,
    bool allow_plain_array = false);

  /// `b[i][j]`, `b[i, j]` or `x = b[i]` over a registered N-D strided view.
  std::optional<exprt>
  try_build_strided_view_index(const nlohmann::json &element);

  /// Total element count and scalar element type of a fixed-shape array of
  /// any rank, or of a contiguous N-D subarray view.
  std::optional<flat_array_shape_info>
  flat_shape_info_of(const exprt &array) const;

  /// Pointer into `array`'s own contiguous storage, `offset` elements in,
  /// that reads as an N-D array of `shape`; registers it for the assignment
  /// target. Declines unless `array` is a fixed-shape array or contiguous view.
  std::optional<exprt> build_contiguous_shaped_view(
    const exprt &array,
    const std::vector<std::size_t> &shape,
    std::size_t offset,
    bool readonly);

  /// `array`'s storage as a pointer to its scalar element type.
  exprt scalar_storage_pointer(const exprt &array, const typet &scalar_ptr_type)
    const;

  /**
   * @brief a.flat[i] = x: builds a dereferenced-pointer lvalue into a's own
   * buffer at flat index i (bounds-checked and negative-index-normalized
   * the same way a registered pointer view's index is), for a fixed-shape
   * 1-D or 2-D array -- the same eligibility and offset/stride=1 math as
   * try_build_ravel_pointer_view, but built inline for an assignment target
   * that has no intermediate bare-name view symbol to register into
   * numpy_pointer_view_info_. Declines (returns std::nullopt) for a
   * non-symbol/non-numpy-tracked source or a source that isn't fixed-shape
   * 1-D/2-D (including 3-D+).
   * @param array      Source 1-D or 2-D array expression.
   * @param index_node Raw AST node for the flat index expression.
   */
  std::optional<exprt> try_build_flat_index_assignment_target(
    const exprt &array,
    const nlohmann::json &index_node);

  /**
   * @brief Lower an N-D mixed slice/index tuple subscript with one or more
   * bounded slice axes and fixed-index axes, e.g. `a[:, 0, 0]`,
   * `a[0:2, 0, 0]`, or `a[:, :, 0]` on a 3-D array. Slice bounds are resolved
   * at conversion time, and the selected scalar cells are copied into a fresh
   * fixed-shape result.
   * @param array      Source N-D array expression.
   * @param idx_nodes  One AST node per axis, in order; at least one must be a
   *                   slice and every non-slice node is treated as a fixed
   *                   index.
   * @param element    The Subscript AST node, used for location info.
   */
  exprt build_mixed_slice_tuple_select(
    const exprt &array,
    const std::vector<nlohmann::json> &idx_nodes,
    const nlohmann::json &element);

  /**
   * @brief Lower integer-array (fancy) indexing `a[[0, 2]]` to a bounded,
   * unrolled sequence of element reads: each entry of @p indices must be a
   * concrete integer literal (or its negation), resolved and bounds-checked
   * at conversion time. @p array must be a fixed-size 1-D or 2-D array; for
   * 2-D arrays the selected rows are copied element-by-element. A 3-D+
   * element type (i.e. @p array itself being 3-D+) is rejected with
   * TypeError, since a deeper row would require a recursive element-wise
   * copy this helper does not implement.
   * @param array   Source fixed-shape (1-D or 2-D) array expression.
   * @param indices AST nodes for each requested index (elts of the List
   *                literal used as the subscript).
   * @param element The Subscript AST node, used for location info.
   */
  exprt build_fancy_index(
    const exprt &array,
    const std::vector<nlohmann::json> &indices,
    const nlohmann::json &element);

  /**
   * @brief Lower a list slice assignment l[lower:upper:step] = value to a
   * __ESBMC_list_slice_assign model call. The step must be a constant literal
   * (or absent); the value must evaluate to a list.
   * @param list_expr  Expression for the target list (value or pointer)
   * @param slice_node The Slice AST node holding lower/upper/step
   * @param value_node The AST node of the assigned (right-hand side) value
   */
  void handle_slice_assignment(
    const exprt &list_expr,
    const nlohmann::json &slice_node,
    const nlohmann::json &value_node);

  exprt compare(const exprt &l1, const exprt &l2, const std::string &op);

  exprt contains(const exprt &item, const exprt &list);

  exprt list_repetition(
    const nlohmann::json &left_node,
    const nlohmann::json &right_node,
    const exprt &lhs,
    const exprt &rhs);

  exprt build_push_list_call(
    const symbolt &list,
    const nlohmann::json &op,
    const exprt &elem,
    bool enable_float_path = true);

  exprt build_insert_list_call(
    const symbolt &list,
    const exprt &index,
    const nlohmann::json &op,
    const exprt &elem);

  exprt build_extend_list_call(
    const symbolt &list,
    const nlohmann::json &op,
    const exprt &other_list);

  // Build: result = lhs + rhs   (concatenation)
  exprt build_concat_list_call(
    const exprt &lhs,
    const exprt &rhs,
    const nlohmann::json &element);

  /// The declaring literal's elements for a variable-held list, or empty when
  /// the literal cannot be used.
  std::vector<exprt> literal_elems_for_variable_list(
    const nlohmann::json *source_node,
    const exprt &source_list);

  /// True when a list's declaring literal still describes its contents.
  bool literal_still_describes_list(
    const exprt &source_list,
    size_t literal_elem_count);

  /// Element type recovered for a bare `list` parameter, or a nil type.
  typet bare_list_param_elem_type(
    const nlohmann::json &param_node,
    const std::string &param_id,
    const typet &annotated);

  /**
   * @brief Create an empty set
   * @return Expression representing the empty set
   */
  exprt get_empty_set();

  /**
   * @brief Convert generator expressions and list comprehensions to lists
   * @param element The GeneratorExp or ListComp AST node
   * @return Expression representing the materialized list
   */
  exprt handle_comprehension(const nlohmann::json &element);

  /**
   * @brief Build a list pop operation
   * @param list The list symbol to pop from
   * @param index The index to pop (default -1 for last element)
   * @param element The AST node for location information
   * @return Expression representing the popped value
   */
  exprt build_pop_list_call(
    const symbolt &list,
    const exprt &index,
    const nlohmann::json &element);

  /**
   * @brief Extract and dereference value from a PyObject* expression
   * @param pyobject_expr Expression representing PyObject* (from list_at or
   * list_pop)
   * @param elem_type The expected element type
   * @param mixed_numeric When true and elem_type is float, the element may be
   *        either an int or a float at runtime (a dynamic index into a mixed
   *        int/float list). The float value is then read by dispatching on the
   *        stored type_id: float elements come from __ESBMC_float_buf, int
   *        elements are promoted from their payload to double.
   * @return Dereferenced value expression (for floats:
   * __ESBMC_float_buf[item->float_idx])
   */
  exprt extract_pyobject_value(
    const exprt &pyobject_expr,
    const typet &elem_type,
    bool mixed_numeric = false,
    bool string_safe = false);

  /**
   * @brief Infer the element type of a list literal AST node, accounting for
   * the int->float promotion applied at construction.
   *
   * A heterogeneous int/float literal is promoted to a homogeneous double list
   * in python_list::get (promote_ints, #5156), so its values all live in
   * __ESBMC_float_buf as doubles. A read of such a literal must therefore use a
   * float element type whatever element the index selects; using the first
   * element's (int) type misreads the stored double's bits (#5160 regression).
   *
   * @return double_type() for a mixed int/float literal, the first element's
   *         type otherwise, or an empty typet() when no element is available.
   */
  typet infer_literal_element_type(const nlohmann::json &list_literal);

  /**
   * @brief Build an inline min/max computation for a mixed int/float list.
   * Accesses each element with its original type, promotes int elements to
   * double for comparison, and returns the winning value as double.
   *
   * Note: Python's min/max returns the winning element in its *original* type
   * (e.g., max([1, 2.5, 3]) returns int 3, not float 3.0). This implementation
   * always returns double, which is correct for float comparisons and equality
   * checks (via float promotion in handle_relational_type_mismatches), but will
   * not work if the result is used as an array index or integer operand.
   *
   * @param list_arg  Expression for the list symbol
   * @param list_id   Symbol identifier of the list
   * @param func_name "min" or "max" (used in error messages)
   * @param comparison_op  exprt::i_gt for max, exprt::i_lt for min
   * @return Expression of type double_type() holding the min/max value
   */
  exprt build_min_max_for_mixed_numeric(
    const exprt &list_arg,
    const std::string &list_id,
    const std::string &func_name,
    irep_idt comparison_op);

  /**
   * @brief Create a list from a range() call
   * @param converter The python converter instance
   * @param range_args The arguments to range() (1-3 arguments: stop, or
   * start+stop, or start+stop+step)
   * @param element The AST node for location information
   * @return Expression representing the list [start, start+step, ..., stop-1]
   * @throws std::runtime_error if range parameters are invalid or too large
   */
  static exprt build_list_from_range(
    python_converter &converter,
    const nlohmann::json &range_args,
    const nlohmann::json &element,
    bool materialise_elements = true);

  /**
   * @brief Materialise a tuple value into a fresh list, pushing each component
   * in order. Used by list(tuple) (and any context that converts a tuple to a
   * list), so the list model sees a real PyListObject* rather than the tuple
   * struct. @p tuple_expr must have tuple struct type.
   */
  static exprt build_list_from_tuple(
    python_converter &converter,
    const exprt &tuple_expr,
    const nlohmann::json &element);

  /**
   * @brief Build a list copy operation
   * @param list The list symbol to copy from
   * @param element The AST node for location information
   * @return Expression representing the copied list
   */
  exprt
  build_copy_list_call(const symbolt &list, const nlohmann::json &element);

  /**
   * @brief Emit a __ESBMC_list_copy_shallow call producing a shallow copy of
   * src_list (Python copy semantics: scalar elements get independent buffers,
   * nested containers stay shared). Used by tuple(list), which must snapshot
   * the source so later list mutations do not show through the tuple.
   * @param src_list List-typed expression (symbol or list-returning call)
   * @param element  AST node used for location info and temp naming
   * @return Symbol expression of the copied list
   */
  exprt
  build_shallow_copy_call(const exprt &src_list, const nlohmann::json &element);

  /**
   * @brief Build a list remove operation (removes first matching element).
   * Raises ValueError (via assertion) if element is not found.
   */
  exprt build_remove_list_call(
    const symbolt &list,
    const nlohmann::json &op,
    const exprt &elem);

  /**
   * @brief Build a list.count(x) call — number of elements equal to x.
   * Returns a size_t-typed value expression.
   */
  exprt build_count_list_call(
    const symbolt &list,
    const nlohmann::json &op,
    const exprt &elem);

  /**
   * @brief Build a list.index(x) call — position of the first element equal to
   * x. Raises ValueError (via assertion) if x is not found. Returns a
   * size_t-typed value expression.
   */
  exprt build_index_list_call(
    const symbolt &list,
    const nlohmann::json &op,
    const exprt &elem);

  /**
   * @brief Build a list.index(x, start[, end]) call — position of the first
   * element equal to x within the slice l[start:end]. start/end follow CPython
   * slice-bound normalization. Raises ValueError (via assertion) if not found.
   */
  exprt build_index_range_list_call(
    const symbolt &list,
    const nlohmann::json &op,
    const exprt &elem,
    const exprt &start,
    const exprt &end);

  /// Shared implementation of build_count_list_call / build_index_list_call;
  /// @p func_id selects the `c:@F@__ESBMC_list_{count,index}` model.
  exprt build_count_index_list_call(
    const symbolt &list,
    const nlohmann::json &op,
    const exprt &elem,
    const std::string &func_id);

  /**
   * @brief Emit a call to a set membership-mutating C model function.
   *
   * Used to implement set.add() and set.discard(): both wrap the same C
   * argument layout (set, &elem, type_id, size) returning bool. The
   * @p method_name selects between "add" and "discard"; the dispatcher
   * resolves it to "__ESBMC_set_add" / "__ESBMC_set_discard".
   */
  exprt build_set_membership_call(
    const symbolt &set,
    const nlohmann::json &op,
    const exprt &elem,
    const std::string &method_name);

  /**
   * @brief The element byte width every recorded element of @p list_id shares,
   *        or 0 when there is no single answer.
   *
   * The list models apply one copy length to every element, so a width is only
   * usable when all of them agree. Scalars and tuples have one, both being
   * stored inline; a pointer-stored element (a nested list, a dict) and mixed
   * widths yield 0, which keeps the model on its symbolic o->size path.
   * Distinct from build_shallow_copy_call, which reads only the last type-map
   * entry.
   */
  BigInt uniform_elem_size(const std::string &list_id) const;

  /** Same, for a list reached as an expression: a non-symbol operand names no
   *  list to look up, so it has no single width and yields 0.
   */
  BigInt uniform_elem_size(const exprt &list) const;

  // True when the list's recorded element types include a tagged scalar, whose
  // payload width is per-element and symbolic (#7716).
  bool has_tagged_elements(const exprt &list) const;

  /// The recorded element type when it is a tagged scalar and the index is not
  /// constant; otherwise the fallback (#7716 family).
  typet tagged_elem_type_or(
    const exprt &array,
    bool constant_index,
    const nlohmann::json &list_node,
    const typet &fallback) const;

  struct shallow_push_call
  {
    const symbolt *func;
    exprt last_arg;
  };

  /** Shallow-push entry point for a copy of `src`. A list of tagged scalars
   *  needs the bounded-copy variant, which reads its trailing argument as a
   *  float_type_id rather than as an element width (#7716).
   */
  shallow_push_call
  select_shallow_push(const exprt &src, const exprt &untagged_last_arg) const;

  shallow_push_call
  select_list_extend(const exprt &src, const exprt &untagged_elem_size) const;

  struct list_eq_target
  {
    const symbolt *func;
    std::vector<exprt> trailing_args;
  };

  /** Equality entry point for `l1 == l2` and the arguments that follow the two
   *  list operands. A tagged element has no single static width and cannot hold
   *  a nested list, so neither elem_size nor the depth stack applies (#7723).
   */
  list_eq_target select_list_eq(
    const exprt &l1,
    const exprt &l2,
    const symbolt &generic_func,
    const std::vector<exprt> &generic_trailing_args) const;

  /**
   * @brief Unpack a list variable into multiple targets, supporting starred
   * expressions.
   *
   * Handles assignments like `a, *b, c = lst` where `lst` is a list variable.
   * Uses __ESBMC_list_at for element access and builds a new list for the
   * starred target.
   *
   * @param ast_node The assignment AST node (for location info and value["id"])
   * @param target   The tuple/list target node containing the target variables
   * @param list_expr The expression representing the source list (pointer type)
   * @param target_block The code block to append assignment instructions to
   */
  void handle_list_var_unpacking(
    const nlohmann::json &ast_node,
    const nlohmann::json &target,
    const exprt &list_expr,
    codet &target_block);

private:
  friend class python_dict_handler;

  // Evaluates value_node once and snapshots it into a fresh temporary, so a
  // multi-write caller (try_build_fill_diagonal_mutation) can reuse the
  // snapshot instead of re-embedding (and re-evaluating) the same expression
  // at each write site. Throws if the value isn't scalar-typed.
  exprt snapshot_scalar_value(const nlohmann::json &value_node);

  // Repeat the elements in `list_elems` `count` times at runtime (`count` may
  // be any integer expression: a constant, a symbol like `n`, or a compound
  // like `m + 1`). Each iteration pushes every element in order. Builds a
  // fresh list so a literal source's element is not reused (avoids off-by-one).
  exprt create_vla(
    const nlohmann::json &element,
    const exprt &count,
    const std::vector<exprt> &list_elems);

  exprt build_list_at_call(
    const exprt &list,
    const exprt &index,
    const nlohmann::json &element);

  list_elem_info
  get_list_element_info(const nlohmann::json &op, const exprt &elem);

  /// Refuses `dict.items()` against a set of tuples, whose pairs the
  /// placeholder view does not model (#7553).
  void reject_items_view_vs_tuple_set(
    const exprt &lhs,
    const exprt &rhs,
    const exprt &converted_lhs,
    const exprt &converted_rhs);

  /// A constructed class instance arrives as a value struct; the element read
  /// expects a reference. Box it so the two agree (#7685).
  exprt as_object_reference(const nlohmann::json &op, const exprt &elem);

  list_elem_info
  get_tagged_element_info(const nlohmann::json &op, const exprt &elem);

  // The type_id a tagged scalar carries when it holds a float, or 0 when the
  // caller opts out of the float path (dict values compare via void*).
  exprt tagged_float_type_id(bool enable_float_path) const;

  symbolt &create_list();

  exprt
  handle_range_slice(const exprt &array, const nlohmann::json &slice_node);

  // Propagate a dict view's element types onto its slice result. Split out for
  // the same reason as normalize_negative_slice_bound below.
  void
  copy_dict_view_elem_types(const exprt &array, const std::string &sliced_id);

  // handle_range_slice's process_bound: normalizes a literal negative bound
  // (-k) to logical_len - k, clamped for an out-of-range k -- see the call
  // site's own comment for why the clamp differs by step direction. Split
  // out to keep handle_range_slice's own decision count from growing
  // further.
  exprt normalize_negative_slice_bound(
    const nlohmann::json &operand_node,
    const exprt &logical_len,
    bool negative_step);

  // Shared core of every ADR-NP-003 scalar pointer view producer; see
  // list_access.cpp for the full rationale, including when this declines.
  std::optional<exprt> build_scalar_pointer_view(
    const exprt &array,
    const typet &elem_type,
    long long offset,
    std::size_t length,
    long long stride,
    bool readonly);

  // ADR-NP-003 etapa 2, first slice: builds a pointer into the base
  // array's own storage for a 1-D, unit-stride, literal-bound slice
  // assigned directly to a bare name, instead of handle_range_slice()'s
  // usual independent copy. Split out to keep that already large
  // function's own decision count from growing further; see
  // list_access.cpp for the full rationale and the cases intentionally
  // left out of this first slice (returns std::nullopt for those, and the
  // caller falls through to the existing copy).
  std::optional<exprt> try_build_1d_pointer_view(
    const exprt &array,
    const typet &elem_type,
    long long step_val,
    bool needs_null_term,
    long long literal_start,
    long long static_slice_len);

  std::optional<exprt> try_build_row_pointer_view(
    const exprt &array,
    const nlohmann::json &slice_node);

  std::optional<exprt> try_build_nd_subarray_pointer_view(
    const exprt &array,
    const nlohmann::json &slice_node);

  std::optional<exprt> try_build_chained_subarray_pointer_view();

  std::optional<exprt> try_build_pointer_array_index(
    const exprt &array,
    const exprt &pos_expr,
    const nlohmann::json &slice_node);

  exprt build_numpy_array_index_access(
    const exprt &array,
    const exprt &pos_expr,
    const nlohmann::json &slice_node);

  std::optional<exprt> try_build_column_pointer_view(
    const exprt &array,
    const nlohmann::json &col_index_node);

  std::optional<exprt> try_copy_numpy_pointer_view_slice(
    const exprt &array,
    const typet &elem_type,
    const nlohmann::json &slice_node,
    long long step_val,
    bool literal_step);

  void emit_slice_zero_step_raise(
    const nlohmann::json &slice_node,
    bool literal_zero_step);

  exprt normalize_and_scale_index(
    const exprt &index,
    long long length,
    long long stride,
    const nlohmann::json &slice_node);

  exprt normalize_and_scale_index(
    const exprt &index,
    const exprt &length,
    const exprt &stride,
    const nlohmann::json &slice_node);

  struct symbolic_slice_params
  {
    exprt offset;
    exprt stride;
    exprt length;
  };

  /// Emits the runtime offset/stride/length of a[lo:hi:st] over a length-n
  /// axis, raising ValueError for st == 0.
  symbolic_slice_params
  emit_symbolic_slice_params(const nlohmann::json &slice_node, long long n);

  /**
   * @brief a[lo:hi:st] with a non-literal step st. Assigned to a bare name it
   * is a pointer view whose offset, length and stride are runtime values
   * computed with CPython's slice rules (a step of zero raises ValueError);
   * anywhere else it is an independent copy. Throws TypeError unless a is a
   * tracked 1-D numpy array and the target is not already a registered view.
   */
  exprt build_symbolic_step_slice(
    const exprt &array,
    const nlohmann::json &slice_node);

  exprt guard_numpy_pointer_view_index(
    const exprt &array,
    const exprt &index,
    const nlohmann::json &slice_node);

  exprt guard_numpy_static_array_index(
    const exprt &array,
    const exprt &index,
    const nlohmann::json &slice_node);

  exprt
  handle_index_access(const exprt &array, const nlohmann::json &slice_node);

  // True when `array` is a 2-D+ numpy array parameter's decayed pointer
  // symbol (register_function_argument's row-pointer decay), i.e. one whose
  // pre-decay shape handle_index_access's own negative-index normalization
  // needs to consult in numpy_param_shapes_. Split out to keep that
  // function's own decision count down.
  bool is_numpy_param_negative_index_target(const exprt &array) const;

  // handle_index_access's own index/negative-index resolution: normalizes
  // pos_expr in place for a literal negative index (a[-1]) against the
  // right size source for `array`'s shape (a numpy parameter's pre-decay
  // shape, an array_typet's own size, or -- when neither applies -- deferred
  // to build_list_at_call's runtime normalization), or sets `index` alone
  // for a compile-time-only type lookup. A no-op for anything but a
  // UnaryOp(USub)/Constant slice. Split out of handle_index_access to keep
  // that function's own decision count down.
  void normalize_index_access_position(
    const exprt &array,
    const nlohmann::json &slice_node,
    const nlohmann::json &list_node,
    exprt &pos_expr,
    size_t &index) const;

  /**
   * @brief Resolve @c array[pos_expr] when the element is itself a list.
   *
   * Sets @p elem_type to the statically recorded element type whenever one
   * exists. Returns the element expression when the nested-list path applies,
   * and nullopt when the caller must fall through to the generic
   * element-type resolution.
   */
  std::optional<exprt> resolve_nested_list_element(
    const exprt &array,
    const exprt &pos_expr,
    size_t index,
    typet &elem_type);

  /**
   * @brief Resolve an index expression against a compile-time-known axis
   * length, normalizing negative values and rejecting out-of-range indices.
   * A literal (constant, or negated constant) index is fully resolved and
   * bounds-checked at conversion time, producing a precise "IndexError:
   * index N is out of bounds for axis A with size L" frontend error. A
   * non-constant (runtime) index is normalized and bounds-checked with an
   * in-model IndexError raise instead, since the concrete value is unknown
   * until symbolic execution.
   * @param idx_node  AST node for the index expression.
   * @param axis_len  Compile-time length of the axis being indexed.
   * @param axis      Axis number, used only in the error message.
   * @param element   AST node used for location info.
   * @return A size_type expression holding the normalized, in-bounds index.
   */
  exprt resolve_fixed_axis_index(
    const nlohmann::json &idx_node,
    const BigInt &axis_len,
    unsigned axis,
    const nlohmann::json &element);

  // Returns (registering if absent) the __python_str_slice symbol:
  //   char* __python_str_slice(const char*, long long, long long, long long)
  const symbolt &get_str_slice_sym();

  exprt remove_function_calls_recursive(exprt &e, const nlohmann::json &node);

  /// The converter-owned registry recording per-instance element types.
  element_type_registry &elem_types();
  const element_type_registry &elem_types() const;

  /**
   * @brief Append every element of src onto dst at runtime.
   *
   * Shared by list concatenation (a + b) and variable-list repetition (lst *
   * n).
   *
   * @param src Source list expression (value or pointer)
   * @param dst Destination list symbol
   * @param element The AST node for location information
   */
  void emit_list_copy(
    const exprt &src,
    const symbolt &dst,
    const nlohmann::json &element);

  /**
   * @brief Handle symbolic (non-constant) range arguments
   * @param converter The python converter instance
   * @param range_args The range arguments from the AST
   * @param element The AST element for location tracking
   * @return Expression representing the symbolic range list
   */
  /**
   * @brief Set symbolic size on a list structure
   * @param converter The python converter instance
   * @param list_expr The list expression to modify
   * @param size_expr The symbolic size expression
   * @param element The AST element for location tracking
   */
  static void set_list_symbolic_size(
    python_converter &converter,
    exprt &list_expr,
    const exprt &size_expr,
    const nlohmann::json &element);

  static exprt handle_symbolic_range(
    python_converter &converter,
    const nlohmann::json &range_args,
    const nlohmann::json &element,
    bool materialise_elements);

  /**
   * @brief Build a concrete range with constant bounds
   * @param converter The python converter instance
   * @param range_args The range arguments from the AST
   * @param element The AST element for location tracking
   * @param arg0 First argument (start or stop depending on arg count)
   * @param arg1 Second argument (stop or step depending on arg count)
   * @param arg2 Third argument (step)
   * @return Expression representing the concrete range list
   * @throws std::runtime_error if range parameters are invalid or too large
   */
  static exprt build_concrete_range(
    python_converter &converter,
    const nlohmann::json &range_args,
    const nlohmann::json &element,
    const std::optional<long long> &arg0,
    const std::optional<long long> &arg1,
    const std::optional<long long> &arg2);

  python_converter &converter_;
  const nlohmann::json &list_value_;
};
