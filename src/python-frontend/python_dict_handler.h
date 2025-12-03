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
 * @section supported_types Supported Value Types
 *
 * The handler supports dictionaries with values of the following types:
 * - Strings (char*)
 * - Integers (signed/unsigned bitvectors)
 * - Booleans
 * - Floating-point numbers
 *
 * @section usage Usage Example
 *
 * @code
 * // Python code:
 * config: dict = {'port': 8080, 'debug': True}
 * x = config['port']
 * assert config['debug'] == True
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
   * @brief Creates a marker expression for a dictionary literal.
   *
   * Returns a placeholder expression that indicates a dictionary literal
   * needs to be created. The actual initialization is deferred to
   * create_dict_from_literal() during assignment handling.
   *
   * @param element The JSON AST node containing the dictionary literal.
   * @return An expression representing the dictionary literal marker.
   * @throws std::runtime_error if the element is not a dictionary literal.
   */
  exprt get_dict_literal(const nlohmann::json &element);

  /**
   * @brief Creates and initializes a dictionary from a literal expression.
   *
   * Generates the GOTO instructions to:
   * 1. Create empty keys and values lists
   * 2. Push all key-value pairs from the literal
   * 3. Assign the lists to the target dictionary struct
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
   * @param dict_expr The expression representing the dictionary.
   * @param slice_node The JSON AST node for the subscript key.
   * @param expected_type The expected type of the value (used for proper
   *        type casting). If not specified, defaults to string (char*).
   *        Supported types: signedbv, unsignedbv, bool, floatbv, string.
   * @return An expression representing the accessed value, properly typed.
   *
   * @note When comparing dictionary values with primitive types (int, bool,
   *       float), the caller should pass the comparison operand's type as
   *       expected_type to ensure proper value extraction.
   */
  exprt handle_dict_subscript(
    const exprt &dict_expr,
    const nlohmann::json &slice_node,
    const typet &expected_type = typet());

  /**
   * @brief Handles dictionary subscript assignment (e.g., `dict['key'] = value`).
   *
   * Generates GOTO instructions to add a new key-value pair to the dictionary.
   * Note: This implementation appends new entries; it does not update existing
   * keys (simplified semantics for verification purposes).
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
   * @return The struct_typet representing Python dictionaries.
   */
  struct_typet get_dict_struct_type();

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
};

#endif // PYTHON_DICT_HANDLER_H