#pragma once

#include <util/c_types.h>
#include <util/arith_tools.h>
#include <util/std_expr.h>
#include <util/std_types.h>
#include <nlohmann/json.hpp>

class python_converter;

// P26: Cached double type — avoids constructing a new floatbv_typet on every call.
// Initialized once on first use; thread-safe (C++11 static local semantics).
inline const typet &cached_double_type()
{
  static const typet instance = double_type();
  return instance;
}

// P4: Static-local struct type — eliminates repeated heap allocations.
// Returns a const reference to the single global instance.
inline const struct_typet &get_complex_struct_type()
{
  static const struct_typet instance = []() {
    struct_typet t;
    t.tag("complex");
    t.components().push_back(
      struct_typet::componentt("real", "real", double_type()));
    t.components().push_back(
      struct_typet::componentt("imag", "imag", double_type()));
    return t;
  }();
  return instance;
}

inline bool is_complex_type(const typet &type)
{
  if (type.id() == "symbol")
    return to_symbol_type(type).get_identifier().as_string() == "tag-complex";

  if (!type.is_struct())
    return false;

  const struct_typet &struct_type = to_struct_type(type);
  const std::string tag = struct_type.tag().as_string();
  return tag == "complex" || tag == "tag-complex";
}

inline exprt make_complex(const exprt &real, const exprt &imag)
{
  struct_exprt complex_expr(get_complex_struct_type());
  const typet &dt = cached_double_type();
  complex_expr.operands().push_back(
    real.type() == dt ? real : typecast_exprt(real, dt));
  complex_expr.operands().push_back(
    imag.type() == dt ? imag : typecast_exprt(imag, dt));
  return complex_expr;
}

inline exprt promote_to_complex(const exprt &value)
{
  if (value.statement() == "cpp-throw")
    return value;

  if (is_complex_type(value.type()))
    return value;

  // Bool cannot be directly typecast to float64 in the SMT encoder.
  // Convert bool → int first; make_complex will then typecast int → double.
  exprt real = value;
  if (real.type().is_bool())
    real = typecast_exprt(real, long_long_int_type());

  return make_complex(real, from_double(0.0, cached_double_type()));
}

inline exprt complex_to_bool_expr(const exprt &complex_expr)
{
  const typet &dt = cached_double_type();
  exprt real = member_exprt(complex_expr, "real", dt);
  exprt imag = member_exprt(complex_expr, "imag", dt);
  exprt zero = from_double(0.0, dt);
  return or_exprt(
    not_exprt(equality_exprt(real, zero)),
    not_exprt(equality_exprt(imag, zero)));
}

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
   * True iff `t`'s storage has no pointer fields (primitives, char arrays,
   * or structs whose components are recursively pointer-free). Used to gate
   * uint64 fast paths in the list operational model — a struct payload with
   * a pointer field cannot be reinterpreted as a uint64 array under
   * ESBMC's byte-encoding.
   */
  bool is_pointer_free(const typet &t) const;

  /*
   * Converts a typet to its string representation.
   * @param t The typet to convert.
   * @return A string containing the type name.
   */
  std::string type_to_string(const typet &t) const;

  /**
   * Returns the Python type name for an ESBMC type, as produced by type().
   * Maps: bool→"bool", floatbv→"float", signedbv/unsignedbv→"int",
   * char array/pointer→"str", complex→"complex", struct→class name.
   */
  std::string get_python_type_name(const typet &t) const;

  /*
   * Returns the detected type for a variable.
   * @param var_name The name of the variable.
   * @return A string representing the variable's type.
   */
  std::string get_var_type(const std::string &var_name) const;
  std::string get_var_classname(const nlohmann::json &value_node) const;

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
