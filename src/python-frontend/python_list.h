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

  python_converter &converter_;
  const nlohmann::json &list_value_;

  // <list_id, <elem_id, elem_type>>
  static std::unordered_map<std::string, TypeInfo> list_type_map;

  exprt remove_function_calls_recursive(exprt &e, const nlohmann::json &node);

  /**
   * @brief Validate and normalize the index expression
   * @param pos_expr The index expression to validate
   * @param slice_node The slice AST node
   * @throws std::runtime_error if index type is invalid (e.g., array indices)
   */
  void validate_index_expression(
    const exprt &pos_expr,
    const nlohmann::json &slice_node) const;

  /**
   * @brief Adjust negative index to positive
   * @param pos_expr The index expression (will be modified if negative)
   * @param slice_node The slice AST node
   * @param array The array being indexed
   * @param list_node The list declaration node (may be null)
   * @return The adjusted index value (for constant indices)
   */
  size_t handle_negative_index(
    exprt &pos_expr,
    const nlohmann::json &slice_node,
    const exprt &array,
    const nlohmann::json &list_node);

  /**
   * @brief Check if this is a nested list access
   * @param array The array being indexed
   * @param index The index value
   * @return Expression for nested list symbol, or nil_exprt if not nested
   */
  exprt try_nested_list_access(const exprt &array, size_t index);

  /**
   * @brief Resolve element type from function parameter annotation
   * @param list_node The parameter node with annotation
   * @return The element type, or empty_typet if not resolvable
   */
  typet resolve_type_from_parameter(const nlohmann::json &list_node) const;

  /**
   * @brief Resolve element type from constant index into list
   * @param array The array being indexed
   * @param index The constant index value
   * @param list_node The list declaration node
   * @return The element type, or empty_typet if not resolvable
   */
  typet resolve_type_from_constant_index(
    const exprt &array,
    size_t index,
    const nlohmann::json &list_node);

  /**
   * @brief Resolve element type from variable index
   * @param array The array being indexed
   * @param list_node The list declaration node
   * @return The element type, or empty_typet if not resolvable
   */
  typet resolve_type_from_variable_index(
    const exprt &array,
    const nlohmann::json &list_node);

  /**
   * @brief Try to resolve element type from dict subscript value
   * @param list_node The variable declaration node
   * @return The element type, or empty_typet if not resolvable
   */
  typet resolve_type_from_dict_subscript(const nlohmann::json &list_node);

  /**
   * @brief Build list access expression and cast to proper type
   * @param array The list/array being indexed
   * @param pos_expr The position expression
   * @param elem_type The element type to cast to
   * @return The constructed access expression
   */
  exprt build_list_access_expression(
    const exprt &array,
    const exprt &pos_expr,
    const typet &elem_type);

  /**
   * @brief Build static array access expression
   * @param array The static array
   * @param pos_expr The position expression
   * @return The index expression
   */
  exprt build_static_array_access(const exprt &array, const exprt &pos_expr);
};
