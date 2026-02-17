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
   * @brief Check if all elements in a list have the same type
   * @param list_id The list identifier
   * @param func_name The function name (for error messages)
   * @return The common element type if all types match, empty typet() otherwise
   * @throws std::runtime_error if mixed types are detected
   */
  static typet check_homogeneous_list_types(
    const std::string &list_id,
    const std::string &func_name);

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
   * @brief Remove the first occurrence of an item from the list type map.
   * Used by handle_list_remove() to keep the type map consistent.
   */
  static void remove_last_type_entry(const std::string &list_id)
  {
    auto it = list_type_map.find(list_id);
    if (it != list_type_map.end() && !it->second.empty())
      it->second.pop_back();
  }

  /**
   * @brief Build a list remove operation (removes first matching element).
   * Raises ValueError (via assertion) if element is not found.
   */
  exprt build_remove_list_call(
    const symbolt &list,
    const nlohmann::json &op,
    const exprt &elem);

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