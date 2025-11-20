#ifndef PYTHON_DICT_HANDLER_H
#define PYTHON_DICT_HANDLER_H

#include <util/std_types.h>
#include <util/expr.h>
#include <nlohmann/json.hpp>

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

class python_converter;
class type_handler;
class contextt;

/**
 * Handler for Python dictionary operations in ESBMC
 * 
 * Dictionaries are modeled as structs with named fields for each key.
 * For example: {'name': 'Bob', 'age': 30} becomes:
 *   struct dict_0 {
 *     char* name;
 *     int age;
 *   };
 */
class python_dict_handler
{
public:
  python_dict_handler(
    python_converter &converter,
    contextt &symbol_table,
    type_handler &type_handler);

  /**
   * Check if a JSON node represents a dictionary literal
   */
  bool is_dict_literal(const nlohmann::json &element) const;

  /**
   * Convert a dictionary literal to a struct expression
   * Creates a struct type with fields for each key and initializes it
   */
  exprt get_dict_literal(const nlohmann::json &element);

  /**
   * Handle dictionary subscript operations (dict[key])
   * Converts to struct member access
   */
  exprt handle_dict_subscript(
    const exprt &dict_expr,
    const nlohmann::json &slice_node);

  /// Mark a dictionary key as deleted
  void mark_key_deleted(const std::string &dict_id, const std::string &key);

  /// Check if a dictionary key has been deleted
  bool is_key_deleted(const std::string &dict_id, const std::string &key) const;

  /// Handle dictionary membership check ("in" / "not in")
  exprt handle_dict_membership(
    const exprt &key_expr,
    const exprt &dict_expr,
    bool negated);

  /// Mark a dictionary key as no longer deleted (when re-assigned)
  void unmark_key_deleted(const std::string &dict_id, const std::string &key);

private:
  // Track deleted dictionary keys: map<dict_id, set<key_name>>
  std::unordered_map<std::string, std::set<std::string>> deleted_keys_;

  python_converter &converter_;
  contextt &symbol_table_;
  type_handler &type_handler_;

  // Counter for generating unique dictionary type names
  static int dict_counter_;

  /**
   * Create a struct type from dictionary keys and value types
   */
  struct_typet create_dict_struct_type(
    const nlohmann::json &dict_node,
    const std::string &dict_name);

  /**
   * Extract string key from a dictionary key node
   */
  std::string extract_dict_key(const nlohmann::json &key_node) const;

  /**
   * Infer the type of a dictionary value expression
   */
  typet infer_value_type(const nlohmann::json &value_node);
};

#endif // PYTHON_DICT_HANDLER_H