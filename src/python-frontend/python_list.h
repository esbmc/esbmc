#pragma once

#include <nlohmann/json.hpp>
#include <util/type.h>
#include <util/expr.h>
#include <util/symbol.h>
#include <set>
#include <utility>

class exprt;
class symbolt;
class python_converter;
class type_handler;

using TypeInfo = std::vector<std::pair<std::string, typet>>;

struct list_elem_info
{
  symbolt *elem_type_sym;
  symbolt *elem_symbol;
  exprt elem_size;
  locationt location;
};

class python_list
{
public:
  python_list(python_converter &converter, const nlohmann::json &list)
    : converter_(converter), list_value_(list)
  {
  }

  static exprt build_split_list(
    python_converter &converter,
    const nlohmann::json &call_node,
    const std::string &input,
    const std::string &separator,
    long long count);

  static exprt build_split_list(
    python_converter &converter,
    const nlohmann::json &call_node,
    const exprt &input_expr,
    const std::string &separator,
    long long count);

  exprt get();

  exprt index(const exprt &array, const nlohmann::json &slice_node);

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
    const exprt &elem);

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

  void add_type_info(
    const std::string &list_symbol_id,
    const std::string &elem_id,
    const typet &elem_type)
  {
    list_type_map[list_symbol_id].push_back(std::make_pair(elem_id, elem_type));
  }

  static void
  copy_type_info(const std::string &source_list, const std::string dest_list)
  {
    if (!list_type_map[source_list].empty())
    {
      list_type_map[dest_list] = list_type_map[source_list];
    }
  }

  /**
   * @brief Record a single element-type entry for a list in the static type map.
   *
   * @param list_symbol_id  Internal symbol identifier of the list.
   * @param elem_id         Symbol identifier of the element, or empty when the
   *                        type is inferred from an annotation rather than from
   *                        a concrete element expression.
   * @param elem_type       ESBMC type of the element.
   */
  static void add_type_info_entry(
    const std::string &list_symbol_id,
    const std::string &elem_id,
    const typet &elem_type)
  {
    list_type_map[list_symbol_id].push_back(std::make_pair(elem_id, elem_type));
  }

  /**
   * @brief Create an empty set
   * @return Expression representing the empty set
   */
  exprt get_empty_set();

  /**
   * Get the element type for a list at a given index.
   * If index is not specified or out of bounds, returns the first element's type.
   * Returns empty typet() if type information is not available.
   */
  static typet
  get_list_element_type(const std::string &list_id, size_t index = 0);

  /**
   * Get the internal symbol id of the element stored at a given index.
   * Returns an empty string when the list or index is not found.
   */
  static std::string
  get_list_element_id(const std::string &list_id, size_t index);

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
   * @param pyobject_expr Expression representing PyObject* (from list_at or list_pop)
   * @param elem_type The expected element type
   * @return Properly cast and dereferenced value expression
   */
  exprt
  extract_pyobject_value(const exprt &pyobject_expr, const typet &elem_type);

  /**
   * @brief Check if all elements in a list have the same type.
   * For mixed int/float lists, returns double_type() (Python promotes int to
   * float for comparisons). Throws for other mixed-type combinations.
   * @param list_id The list identifier
   * @param func_name The function name (for error messages)
   * @return The common element type, double_type() for int/float mix, or
   *         empty typet() when the list is unknown.
   * @throws std::runtime_error if incompatible mixed types are detected
   */
  static typet check_homogeneous_list_types(
    const std::string &list_id,
    const std::string &func_name);

  /**
   * @brief Return true when the list contains both integer and float elements.
   * Used to detect mixed-numeric lists that need special handling in min/max.
   */
  static bool has_mixed_numeric_types(const std::string &list_id);

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
   * @param range_args The arguments to range() (1-3 arguments: stop, or start+stop, or start+stop+step)
   * @param element The AST node for location information
   * @return Expression representing the list [start, start+step, ..., stop-1]
   * @throws std::runtime_error if range parameters are invalid or too large
   */
  static exprt build_list_from_range(
    python_converter &converter,
    const nlohmann::json &range_args,
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
   * @brief Build a list remove operation (removes first matching element).
   * Raises ValueError (via assertion) if element is not found.
   */
  exprt build_remove_list_call(
    const symbolt &list,
    const nlohmann::json &op,
    const exprt &elem);

  /**
   * @brief Return the number of type entries recorded for a list.
   *
   * Provides a safe, bounded count of the elements stored in list_type_map
   * for the given list identifier.
   *
   * @param list_id  The internal symbol identifier of the list (e.g.
   *                 "c:main.py@42@F@main@lst").
   * @return  Number of type entries in the map for this list, or 0 if the
   *          list is unknown or was constructed with no recorded elements.
   */
  static size_t get_list_type_map_size(const std::string &list_id);

  /** Compute the type_flag and float_type_id for a list, using the same
   *  encoding as __ESBMC_list_sort and __ESBMC_list_lt:
   *    0 = all-integer, 1 = all-float, 2 = string, 3 = mixed int+float.
   *  Only examines the element types recorded in list_type_map for list_id.
   *  Note: currently only inspects a single list; for mixed-type comparisons
   *  (e.g. int list vs float list) the caller should merge flags from both
   *  operands. */
  static void get_list_type_flags(
    const std::string &list_id,
    const type_handler &th,
    int &type_flag,
    size_t &float_type_id);

  /**
   * @brief Reverse the compile-time type-info vector for a list.
   *
   * Mirrors the runtime element reordering performed by
   * __ESBMC_list_reverse, so that subsequent index-based type lookups
   * (e.g. list[0]) continue to resolve to the correct element type
   * after an in-place reversal.
   *
   * Has no effect if the list is unknown, empty, or contains only one
   * element (those cases are already trivially reversed).
   *
   * @param list_id  The internal symbol identifier of the list (e.g.
   *                 "c:main.py@42@F@main@lst").
   */
  static void reverse_type_info(const std::string &list_id);

private:
  friend class python_dict_handler;

  exprt create_vla(
    const nlohmann::json &element,
    const symbolt *list,
    symbolt *size_var,
    const exprt &list_elem);

  exprt build_list_at_call(
    const exprt &list,
    const exprt &index,
    const nlohmann::json &element);

  list_elem_info
  get_list_element_info(const nlohmann::json &op, const exprt &elem);

  symbolt &create_list();

  exprt
  handle_range_slice(const exprt &array, const nlohmann::json &slice_node);

  exprt
  handle_index_access(const exprt &array, const nlohmann::json &slice_node);

  exprt remove_function_calls_recursive(exprt &e, const nlohmann::json &node);

  void copy_type_map_entries(
    const std::string &from_list_id,
    const std::string &to_list_id);

  /**
   * @brief Handle symbolic (non-constant) range arguments
   * @param converter The python converter instance
   * @param range_args The range arguments from the AST
   * @param element The AST element for location tracking
   * @return Expression representing the symbolic range list
   */
  static exprt handle_symbolic_range(
    python_converter &converter,
    const nlohmann::json &range_args,
    const nlohmann::json &element);

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

  // <list_id, <elem_id, elem_type>>
  static std::unordered_map<std::string, TypeInfo> list_type_map;
};