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

  /**
   * @brief Find the list node declaration for type information
   */
  nlohmann::json find_list_node_declaration();

  /**
   * @brief Validate that the index is not an array type
   */
  void validate_index_type(const exprt &index_expr);

  /**
   * @brief Process negative index by converting it to positive index
   * @return Pair of processed index expression and actual index value
   */
  std::pair<exprt, size_t> process_negative_index(
    const nlohmann::json &slice_node,
    const nlohmann::json &list_node,
    const exprt &array,
    exprt index_expr);

  /**
   * @brief Check if this is a nested list access and handle it
   * @return The nested list symbol expression, or empty if not nested
   */
  exprt handle_nested_list_access(const exprt &array, size_t index);

  /**
   * @brief Get element type from function parameter annotation
   */
  typet get_element_type_from_parameter(const nlohmann::json &list_node);

  /**
   * @brief Get element type from list type map or annotation fallback
   */
  typet get_element_type_from_type_map(
    const exprt &array,
    const nlohmann::json &list_node,
    size_t index);

  /**
   * @brief Handle out-of-bounds access with appropriate error or type inference
   */
  typet handle_out_of_bounds_access(
    const nlohmann::json &slice_node,
    const nlohmann::json &list_node);

  /**
   * @brief Get element type for variable-based indexing
   */
  typet get_element_type_for_variable_index(
    const exprt &array,
    nlohmann::json list_node);

  /**
   * @brief Navigate through nested declarations to find element type
   */
  typet get_element_type_from_nested_declaration(nlohmann::json list_node);

  /**
   * @brief Get element type from dictionary subscript access (e.g., d['key'])
   */
  typet get_element_type_from_dict_subscript(const nlohmann::json &list_node);

  /**
   * @brief Determine element type for list-like symbol access
   */
  typet determine_element_type_for_list_access(
    const exprt &array,
    const nlohmann::json &slice_node,
    const nlohmann::json &list_node,
    size_t index);

  /**
   * @brief Build the final expression for accessing a list element
   */
  exprt build_list_element_expression(
    const exprt &array,
    const exprt &index_expr,
    const typet &elem_type);

  /**
   * @brief Handle static string indexing with bounds checking
   */
  exprt
  handle_static_string_indexing(const exprt &array, const exprt &index_expr);

  /**
   * @brief Handle static array indexing
   */
  exprt
  handle_static_array_indexing(const exprt &array, const exprt &index_expr);

  python_converter &converter_;
  const nlohmann::json &list_value_;

  // <list_id, <elem_id, elem_type>>
  static std::unordered_map<std::string, TypeInfo> list_type_map;
};
