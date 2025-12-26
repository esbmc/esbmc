/**
 * @file python_dict_handler.h
 * @brief Handler for Python dictionary operations in ESBMC's Python frontend.
 *
 * This module provides support for converting Python dictionary constructs
 * into ESBMC's intermediate representation (GOTO programs). It handles
 * dictionary literals, subscript access, assignment, membership testing,
 * and deletion operations.
 *
 * @section dict_representation Dictionary Representation
 *
 * Python dictionaries are represented as a struct containing two parallel
 * lists: one for keys and one for values. The struct type is defined as:
 *
 * @code
 * struct __python_dict__ {
 *   PyListObject* keys;    // List of dictionary keys
 *   PyListObject* values;  // List of dictionary values (parallel to keys)
 * };
 * @endcode
 *
 * Dictionary values are stored as PyObject instances with type information
 * (type_id) that allows runtime type identification for proper value extraction.
 *
 * @section nested_dict_support Nested Dictionary Support
 *
 * Nested dictionaries (e.g., `dict[int, dict[int, int]]`) require special
 * handling because dict structs contain pointer fields. Instead of copying
 * the entire nested dict struct, we store a **pointer** to the dict.
 *
 * **Memory Model:**
 * @code
 * Parent Dict:
 *   values_list -> [ptr1, ptr2, ...]
 *                   ↓
 *               [Points to nested dict struct]
 * @endcode
 *
 * **Type Safety:**
 * Each nested dict type gets a unique hash based on its full type structure,
 * ensuring `dict[int, dict[int, int]]` has a different hash than
 * `dict[str, dict[str, str]]`. This prevents type confusion during retrieval.
 *
 * @section supported_types Supported Value Types
 *
 * The handler supports dictionaries with values of the following types:
 * - Strings (char*)
 * - Integers (signed/unsigned bitvectors)
 * - Booleans
 * - Floating-point numbers
 * - Lists
 * - Nested dictionaries (with pointer-based storage)
 *
 * @section usage Usage Example
 *
 * @code
 * // Python code:
 * config: dict = {'port': 8080, 'debug': True}
 * x = config['port']
 * assert config['debug'] == True
 *
 * // Nested dict example:
 * nested: dict[int, dict[int, int]] = {1: {3: 4}}
 * value = nested[1][3]
 * assert value == 4
 * @endcode
 *
 * @see python_converter
 * @see python_list
 * @see type_handler
 */

#ifndef PYTHON_DICT_HANDLER_H
#define PYTHON_DICT_HANDLER_H

#include <util/std_types.h>
#include <util/std_code.h>
#include <util/expr.h>
#include <nlohmann/json.hpp>

#include <string>

class python_converter;
class type_handler;
class contextt;
class symbolt;

/**
 * @class python_dict_handler
 * @brief Handles Python dictionary operations for ESBMC verification.
 *
 * This class is responsible for converting Python dictionary operations
 * into equivalent C/GOTO constructs that can be processed by ESBMC's
 * symbolic execution engine.
 *
 * The handler works in conjunction with python_converter to process
 * dictionary-related AST nodes from the Python frontend.
 *
 * **Key Features:**
 * - Dictionary literal creation
 * - Subscript access with type-safe value retrieval
 * - Dictionary mutation (assignment, deletion)
 * - Membership testing (`in` / `not in`)
 * - Nested dictionary support with proper reference semantics
 * - Type resolution for dictionary subscripts
 */
class python_dict_handler
{
public:
  /**
   * @brief Constructs a new dictionary handler.
   *
   * @param converter Reference to the main Python converter for expression handling.
   * @param symbol_table Reference to the symbol table (context) for symbol management.
   * @param type_handler Reference to the type handler for type operations.
   */
  python_dict_handler(
    python_converter &converter,
    contextt &symbol_table,
    type_handler &type_handler);

  /**
   * @brief Checks if a JSON AST node represents a dictionary literal.
   *
   * Determines whether the given AST element is a Python dictionary
   * literal (e.g., `{'key': 'value'}`).
   *
   * @param element The JSON AST node to check.
   * @return true if the element is a dictionary literal, false otherwise.
   */
  bool is_dict_literal(const nlohmann::json &element) const;

  /**
   * @brief Creates a dictionary literal expression.
   *
   * Returns a symbol expression for a dictionary literal. For nested
   * dictionaries, this creates a temporary variable to ensure the dict
   * exists as a concrete symbol that can be referenced via pointer.
   *
   * The actual initialization is performed by create_dict_from_literal().
   *
   * @param element The JSON AST node containing the dictionary literal.
   * @return A symbol expression representing the initialized dictionary.
   * @throws std::runtime_error if the element is not a dictionary literal.
   */
  exprt get_dict_literal(const nlohmann::json &element);

  /**
   * @brief Creates and initializes a dictionary from a literal expression.
   *
   * Generates the GOTO instructions to:
   * 1. Create empty keys and values lists
   * 2. Push all key-value pairs from the literal
   *    - For nested dicts: stores pointer to dict (reference semantics)
   *    - For regular values: stores value directly (value semantics)
   * 3. Assign the lists to the target dictionary struct
   *
   * **Nested Dictionary Handling:**
   * When a value is a nested dict, the implementation:
   * - Creates a pointer to the dict struct
   * - Generates a unique type hash for the specific dict type
   * - Stores the pointer (8 bytes) instead of copying the struct
   *
   * @param element The JSON AST node containing the dictionary literal.
   * @param target_symbol The LHS expression (dictionary variable) to assign to.
   * @return The target symbol expression after initialization.
   */
  exprt create_dict_from_literal(
    const nlohmann::json &element,
    const exprt &target_symbol);

  /**
   * @brief Handles dictionary subscript access (e.g., `dict['key']`).
   *
   * Generates GOTO instructions to:
   * 1. Find the index of the key in the keys list
   * 2. Retrieve the value at that index from the values list
   * 3. Extract and cast the value based on the expected type
   *
   * **Type-Safe Retrieval:**
   * - For nested dicts: reconstructs dict reference from stored pointer
   * - For primitive types: casts void* to appropriate type
   * - Uses expected_type to ensure correct casting
   *
   * @param dict_expr The expression representing the dictionary.
   * @param slice_node The JSON AST node for the subscript key.
   * @param expected_type The expected type of the value (used for proper
   *        type casting). If not specified, defaults to string (char*).
   *        Supported types: signedbv, unsignedbv, bool, floatbv, string,
   *        list, and nested dict types.
   * @return An expression representing the accessed value, properly typed.
   *
   * @note When comparing dictionary values with primitive types (int, bool,
   *       float), the caller should pass the comparison operand's type as
   *       expected_type to ensure proper value extraction.
   *
   * @note For nested dicts, the expected_type must be the complete dict type.
   *       This enables type-safe retrieval with proper hash validation.
   */
  exprt handle_dict_subscript(
    const exprt &dict_expr,
    const nlohmann::json &slice_node,
    const typet &expected_type = typet());

  /**
   * @brief Handles dictionary subscript assignment (e.g., `dict['key'] = value`).
   *
   * Generates GOTO instructions to update or insert a key-value pair:
   * - If key exists: updates the existing value at that index
   * - If key doesn't exist: appends new key-value pair
   *
   * This provides proper Python dictionary semantics where assignment
   * either updates or creates entries.
   *
   * @param dict_expr The expression representing the dictionary.
   * @param slice_node The JSON AST node for the subscript key.
   * @param value The expression representing the value to assign.
   * @param target_block The code block to append generated instructions to.
   */
  void handle_dict_subscript_assign(
    const exprt &dict_expr,
    const nlohmann::json &slice_node,
    const exprt &value,
    codet &target_block);

  /**
   * @brief Handles dictionary membership testing (`in` / `not in` operators).
   *
   * Checks whether a key exists in the dictionary's keys list.
   *
   * @param key_expr The expression representing the key to search for.
   * @param dict_expr The expression representing the dictionary.
   * @param negated If true, implements `not in`; if false, implements `in`.
   * @return A boolean expression representing the membership test result.
   *
   * @code
   * // Python: 'key' in my_dict
   * // Python: 'key' not in my_dict
   * @endcode
   */
  exprt handle_dict_membership(
    const exprt &key_expr,
    const exprt &dict_expr,
    bool negated);

  /**
   * @brief Handles dictionary key deletion (e.g., `del dict['key']`).
   *
   * Generates GOTO instructions to:
   * - Check if key exists (membership test)
   * - If exists: find index and remove from both keys and values lists
   * - If not exists: raise KeyError exception
   *
   * This provides proper Python semantics where deleting a non-existent
   * key raises an error.
   *
   * @param dict_expr The expression representing the dictionary.
   * @param slice_node The JSON AST node for the key to delete.
   * @param target_block The code block to append generated instructions to.
   */
  void handle_dict_delete(
    const exprt &dict_expr,
    const nlohmann::json &slice_node,
    codet &target_block);

  /**
   * @brief Checks if a type represents a Python dictionary.
   *
   * @param type The type to check.
   * @return true if the type is the dictionary struct type, false otherwise.
   */
  bool is_dict_type(const typet &type) const;

  /**
   * @brief Gets or creates the dictionary struct type.
   *
   * Returns the struct type used to represent Python dictionaries.
   * If the type doesn't exist in the symbol table, it is created and added.
   *
   * The dict struct contains:
   * - `keys`: PyListObject* - list of dictionary keys
   * - `values`: PyListObject* - list of dictionary values
   *
   * @return The struct_typet representing Python dictionaries.
   */
  struct_typet get_dict_struct_type();

  /**
   * @brief Resolve dictionary subscript types for comparisons
   *
   * When comparing dict[key] with primitives, this method ensures both
   * operands have compatible types by dereferencing dict subscripts.
   *
   * Handles three cases:
   * 1. LHS is dict subscript, RHS is primitive
   * 2. RHS is dict subscript, LHS is primitive  
   * 3. Both are dict subscripts
   *
   * @param left Left JSON AST node
   * @param right Right JSON AST node
   * @param lhs Left operand expression (modified in place)
   * @param rhs Right operand expression (modified in place)
   */
  void resolve_dict_subscript_types(
    const nlohmann::json &left,
    const nlohmann::json &right,
    exprt &lhs,
    exprt &rhs);

  /**
   * @brief Extract value type from dict type annotation
   *
   * For annotations like `dict[K, V]`, extracts the value type V.
   *
   * @param annotation_node The annotation AST node (Subscript with slice)
   * @return The value type, or empty_typet if cannot be determined
   */
  typet
  get_dict_value_type_from_annotation(const nlohmann::json &annotation_node);

  /**
   * @brief Resolve expected type for dict subscript using variable annotation
   *
   * Looks up the dict variable's annotation to determine what type
   * the subscript operation should return.
   *
   * @param dict_expr The dictionary expression (must be a symbol)
   * @return The expected value type from annotation, or empty_typet
   */
  typet resolve_expected_type_for_dict_subscript(const exprt &dict_expr);

private:
  /// Reference to the main Python converter
  python_converter &converter_;

  /// Reference to the symbol table for symbol management
  contextt &symbol_table_;

  /// Reference to the type handler for type operations
  type_handler &type_handler_;

  /// Counter for generating unique dictionary variable names
  static int dict_counter_;

  /**
   * @brief Extracts the key expression from a subscript slice node.
   *
   * @param slice_node The JSON AST node representing the subscript slice.
   * @return The expression representing the key.
   */
  exprt get_key_expr(const nlohmann::json &slice_node);

  /**
   * @brief Generate a unique type hash for nested dictionary types
   * Creates a stable hash based on the full type structure including key/value types.
   * This ensures type safety when retrieving nested dicts from the generic list.
   * @param dict_type The dictionary type to hash
   * @return A unique hash value for this specific dict type
   */
  static size_t generate_nested_dict_type_hash(const typet &dict_type);

  /**
   * @brief Safely cast void* through pointer_type to dict pointer type
   * Performs the multi-step casting required to convert stored pointer
   * values back to typed dict pointers.
   * @param node AST node for context
   * @param obj_value The void* value from storage
   * @param target_ptr_type The target pointer type
   * @param location Source location
   * @return Expression of target_ptr_type
   */
  exprt safe_cast_to_dict_pointer(
    const nlohmann::json &node,
    const exprt &obj_value,
    const typet &target_ptr_type,
    const locationt &location);

  /**
   * @brief Store a nested dictionary value in the values list
   * @param element AST node for the dict element
   * @param values_list Symbol for the values list
   * @param value_expr Expression for the nested dict
   * @param location Source location
   */
  void store_nested_dict_value(
    const nlohmann::json &element,
    const symbolt &values_list,
    const exprt &value_expr,
    const locationt &location);

  /**
   * @brief Retrieve a nested dictionary from the values list
   * @param slice_node AST node for the slice operation
   * @param obj_value The stored pointer value
   * @param expected_type The expected dict type
   * @param location Source location
   * @return Expression referencing the dict struct
   */
  exprt retrieve_nested_dict_value(
    const nlohmann::json &slice_node,
    const exprt &obj_value,
    const typet &expected_type,
    const locationt &location);

  /**
   * @brief Generate a unique dictionary name based on source location
   * 
   * Creates deterministic names using file, line, and column information.
   * Falls back to JSON node hash if location is unavailable.
   * 
   * @param element The JSON AST node for the dictionary
   * @param location The source location
   * @return A unique dictionary identifier
   */
  std::string generate_unique_dict_name(
    const nlohmann::json &element,
    const locationt &location) const;
};

#endif // PYTHON_DICT_HANDLER_H