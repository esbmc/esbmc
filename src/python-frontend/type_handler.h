#pragma once

#include <util/c_types.h>
#include <nlohmann/json.hpp>

class python_converter;

class type_handler
{
public:
  type_handler(const python_converter &converter);

  /*
   * Checks if the AST node represents a constructor call.
   * @param json AST node in JSON format corresponding to a function call.
   * @return true if the node is a constructor call, false otherwise.
  */
  bool is_constructor_call(const nlohmann::json &json) const;

  /*
   * Converts a typet to its string representation.
   * @param t The typet to convert.
   * @return A string containing the type name.
   */
  std::string type_to_string(const typet &t) const;

  /*
   * Returns the detected type for a variable.
   * @param var_name The name of the variable.
   * @return A string representing the variable's type.
   */
  std::string get_var_type(const std::string &var_name) const;

  /*
   * Creates an array_typet.
   * @param sub_type The type of elements in the array.
   * @param size The number of elements in the array.
   * @return The constructed array typet.
   */
  typet build_array(const typet &sub_type, const size_t size) const;

  std::vector<int> get_array_type_shape(const typet &array_type) const;

  /*
   * Creates a typet based on a Python type.
   * @param ast_type The name of the Python type (e.g., "int", "str").
   * @param type_size The size used for container types like arrays and lists (default is 0).
   * @return The corresponding typet.
   */
  typet get_typet(const std::string &ast_type, size_t type_size = 0) const;

  /*
   * Creates a typet directly from a JSON value.
   * @param elem A JSON node representing a value.
   * @return The corresponding typet.
   */
  typet get_typet(const nlohmann::json &elem) const;

  /*
   * Checks if a container contains elements of different types.
   * @param container The JSON node representing the container.
   * @return true if the container contains multiple types, false otherwise.
   */
  bool has_multiple_types(const nlohmann::json &container) const;

  /*
   * Builds an array_typet from a list of JSON elements by detecting the elements' subtypes and size.
   * @param list_value The list of elements of an array.
   * @return The array_typet capable of holding the list's values.
   */
  typet get_list_type(const nlohmann::json &list_value) const;

  const typet get_list_type() const;

  typet get_list_element_type() const;

  /*
   * Gets the generic dictionary type from the symbol table.
   * @return A pointer to the generic __python_dict__ struct type.
   */
  const typet get_dict_type() const;

  /*
   * Infers the specific dictionary type from a JSON value.
   * @param dict_value The JSON node representing the dict value.
   * @return The inferred dictionary type based on the value's structure.
   */
  typet get_dict_type(const nlohmann::json &dict_value) const;

  typet get_tuple_type(const nlohmann::json &tuple_node) const;

  /*
   * Determines the type of an operand in binary operations.
   * @param operand The JSON node representing the operand.
   * @return A string representing the operand's type.
   */
  std::string get_operand_type(const nlohmann::json &operand) const;

  /*
   * Checks whether the given JSON object represents a 2D array (list of lists).
   * @param arr The JSON object to check.
   * @return true if it's a 2D array, false otherwise.
   */
  bool is_2d_array(const nlohmann::json &arr) const;

  int get_array_dimensions(const nlohmann::json &arr) const;

  /*
   * Determines the numeric width (in bits) of a given type.
   * @param type The type object to analyze for width determination.
   * @return The width of the type in bits as a size_t value.
   */
  size_t get_type_width(const typet &type) const;

  typet build_optional_type(const typet &base_type);

  /*
   * Returns true if `class_name` is the same as, or derives (directly or
   * indirectly) from, `expected_base` in the current AST.
   */
  bool class_derives_from(
    const std::string &class_name,
    const std::string &expected_base) const;

private:
  /// Encapsulate the const_cast in one place with clear documentation
  exprt get_expr_helper(const nlohmann::json &json) const;

  /// Check if two types are compatible for list homogeneity checking
  bool are_types_compatible(const typet &t1, const typet &t2) const;

  /// Get a normalized/canonical type for list element type inference
  typet get_canonical_string_type(const typet &t) const;

  /// Resolves a Call's func node (id or Attribute) to a typet
  typet get_typet_from_call_func(const nlohmann::json &func) const;

  const python_converter &converter_;
};
