#ifndef ESBMC_PYTHON_TUPLE_HANDLER_H
#define ESBMC_PYTHON_TUPLE_HANDLER_H

#include <python-frontend/python_converter.h>
#include <util/type.h>
#include <util/expr.h>
#include <nlohmann/json.hpp>
#include <string>

class python_converter;
class type_handler;

/**
 * @brief Handler for Python tuple operations
 * 
 * This class manages tuple creation, subscripting, unpacking, and type handling
 * for Python tuples in the ESBMC converter.
 */
class tuple_handler
{
public:
  /**
   * @brief Construct a new tuple handler
   * @param converter Reference to the parent python_converter
   * @param type_handler Reference to the type_handler for type operations
   */
  tuple_handler(python_converter &converter, type_handler &type_handler);

  /**
   * @brief Create a tuple expression from AST elements
   * @param element JSON AST node representing a Tuple
   * @return exprt The tuple represented as a struct expression
   */
  exprt get_tuple_expr(const nlohmann::json &element);

  /**
   * @brief Handle tuple subscripting (e.g., t[0])
   * @param array The tuple expression to subscript
   * @param slice The index expression
   * @param element The original AST element for location info
   * @return exprt Member access expression for the tuple element
   */
  exprt handle_tuple_subscript(
    const exprt &array,
    const nlohmann::json &slice,
    const nlohmann::json &element);

  /**
   * @brief Check if a type represents a tuple
   * @param type The type to check
   * @return bool True if the type is a tuple struct type
   */
  bool is_tuple_type(const typet &type) const;

  /**
   * @brief Handle tuple unpacking assignments
   * @param ast_node The assignment AST node
   * @param target The target tuple pattern (e.g., (x, y, z))
   * @param rhs The right-hand side expression
   * @param target_block The code block to append assignments to
   */
  void handle_tuple_unpacking(
    const nlohmann::json &ast_node,
    const nlohmann::json &target,
    exprt &rhs,
    codet &target_block);

  /**
   * @brief Get tuple type from annotation (e.g., tuple[int, str])
   * @param annotation_node The annotation AST node
   * @return typet The tuple struct type
   */
  typet get_tuple_type_from_annotation(const nlohmann::json &annotation_node);

  /**
   * @brief Prepare RHS for unpacking (handle function calls)
   * @param ast_node The assignment AST node
   * @param rhs The right-hand side expression
   * @param target_block The code block to append temporary declarations to
   * @return exprt The prepared expression (original or temporary variable)
   */
  exprt prepare_rhs_for_unpacking(
    const nlohmann::json &ast_node,
    exprt &rhs,
    codet &target_block);

private:
  /**
   * @brief Build a unique tag name for a tuple based on element types
   * @param element_types Vector of types for tuple elements
   * @return std::string The tag name (e.g., "tag-tuple_int_str")
   */
  std::string build_tuple_tag(const std::vector<typet> &element_types) const;

  /**
   * @brief Create a tuple struct type from element types
   * @param element_types Vector of types for tuple elements
   * @return struct_typet The tuple struct type with components
   */
  struct_typet
  create_tuple_struct_type(const std::vector<typet> &element_types) const;

  python_converter &converter_;
  type_handler &type_handler_;
};

#endif // ESBMC_PYTHON_TUPLE_HANDLER_H