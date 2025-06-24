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

<<<<<<< HEAD
<<<<<<< HEAD
  int get_array_dimensions(const nlohmann::json &arr) const;

=======
>>>>>>> 7b9925f49 ([python] add support for numpy.dot() with 1D and mixed-dimension arrays (#2489))
=======
  int get_array_dimensions(const nlohmann::json &arr) const;

>>>>>>> 468bb9ad1 ([numpy] disallow 3D or higher-dimensional arrays (#2492))
private:
  const python_converter &converter_;
};
