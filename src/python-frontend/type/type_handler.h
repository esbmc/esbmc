#pragma once

#include <util/c_types.h>
#include <util/arith_tools.h>
#include <util/std_expr.h>
#include <util/std_types.h>
#include <util/migrate.h>
#include <irep2/irep2_utils.h>
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
  // V.3: build the complex struct literal in IREP2, back-migrating once. Each
  // non-double operand's int/bool/float -> double conversion is exactly the
  // typecast2t migrate_expr builds for the legacy typecast_exprt (ieee cast,
  // rounding mode implicit), so the back-migrated struct is byte-identical to
  // the old struct_exprt. Mirrors complex_typecast / complex_to_bool_expr.
  const typet &dt = cached_double_type();
  const type2tc dt2 = migrate_type(dt);

  auto to_double2 = [&](const exprt &v) -> expr2tc {
    expr2tc v2;
    migrate_expr(v, v2);
    return v.type() == dt ? v2 : typecast2tc(dt2, v2);
  };

  std::vector<expr2tc> members{to_double2(real), to_double2(imag)};
  return migrate_expr_back(
    constant_struct2tc(migrate_type(get_complex_struct_type()), members));
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
  // V.3: build `z.real != 0.0 || z.imag != 0.0` in IREP2, back-migrating once.
  // member2t over a complex source is exactly the node migrate_expr builds for
  // the legacy member access at goto-convert (util/migrate.cpp:1580), and the
  // not(equal) shape mirrors the legacy or_exprt(not_exprt(equality_exprt(...)))
  // verbatim, so the back-migrated tree is byte-identical to the old one.
  const type2tc dt2 = migrate_type(dt);
  expr2tc complex2;
  migrate_expr(complex_expr, complex2);
  expr2tc zero2;
  migrate_expr(from_double(0.0, dt), zero2);

  expr2tc real_nz =
    not2tc(equality2tc(member2tc(dt2, complex2, "real"), zero2));
  expr2tc imag_nz =
    not2tc(equality2tc(member2tc(dt2, complex2, "imag"), zero2));
  return migrate_expr_back(or2tc(real_nz, imag_nz));
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

  /**
   * Lowering for Python `int` — arbitrarily large in the language spec.
   * Default (bitvector backends): int64 approximation via long_long_int_type().
   * Under --ir (int-encoding=true): a wide signed bitvector. SMT layer strips
   * the width at conversion time, giving true unbounded semantics; the width
   * exists only to keep the IR self-consistent and to size constant-fold
   * intermediate values. See issue #4642.
   */
  static typet python_int_typet();

  /// Bit width of python_int_typet() — exposed for callsites that need to
  /// compute representable ranges (e.g. constant-fold exponent caps).
  static unsigned python_int_width();

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

  /**
   * @brief Returns the registered struct type used for Python slice objects.
   *
   * The struct is defined in `c2goto/library/python/python_types.h` as
   * `__ESBMC_PySliceObj`. The frontend constructs values of this type when
   * lowering `Slice` AST nodes and the `slice()` builtin.
   */
  typet get_slice_type() const;

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
