#pragma once

#include <python-frontend/global_scope.h>
#include <util/context.h>
#include <util/namespace.h>
#include <nlohmann/json.hpp>

#include <map>
#include <set>

class codet;
class struct_typet;
class function_id;
class symbol_id;

class python_converter
{
public:
  python_converter(
    contextt &_context,
    const nlohmann::json &ast,
    const global_scope &gs);
  void convert();

private:
  void get_var_assign(const nlohmann::json &ast_node, codet &target_block);
  void get_compound_assign(const nlohmann::json &ast_node, codet &target_block);
  void
  get_return_statements(const nlohmann::json &ast_node, codet &target_block);
  void get_function_definition(const nlohmann::json &function_node);
  void
  get_class_definition(const nlohmann::json &class_node, codet &target_block);
  std::string get_operand_type(const nlohmann::json &element);

  locationt get_location_from_decl(const nlohmann::json &ast_node);
  exprt get_expr(const nlohmann::json &element);
  exprt get_unary_operator_expr(const nlohmann::json &element);
  exprt get_binary_operator_expr(const nlohmann::json &element);
  exprt get_logical_operator_expr(const nlohmann::json &element);
  exprt get_conditional_stm(const nlohmann::json &ast_node);
  exprt get_function_call(const nlohmann::json &ast_block);
  exprt get_literal(const nlohmann::json &element);
  exprt get_block(const nlohmann::json &ast_block);

  bool has_multiple_types(const nlohmann::json &container);
  void adjust_statement_types(exprt &lhs, exprt &rhs) const;

  symbol_id build_function_id(const nlohmann::json &element);
  symbol_id create_symbol_id() const;
  symbol_id create_symbol_id(const std::string &filename) const;

  bool is_constructor_call(const nlohmann::json &json);
  typet get_typet(const std::string &ast_type, size_t type_size = 0);
  typet get_typet(const nlohmann::json &elem);
  std::string get_var_type(const std::string &var_name) const;
  typet get_list_type(const nlohmann::json &list);
  void get_attributes_from_self(
    const nlohmann::json &method_body,
    struct_typet &clazz);

  symbolt *find_function_in_base_classes(
    const std::string &class_name,
    const std::string &symbol_id,
    std::string method_name,
    bool is_ctor) const;

  symbolt *find_symbol_in_imported_modules(const std::string &symbol_id) const;

  symbolt *find_symbol_in_global_scope(std::string &symbol_id) const;

  void update_instance_from_self(
    const std::string &class_name,
    const std::string &func_name,
    const std::string &obj_symbol_id);

  std::string get_classname_from_symbol_id(const std::string &symbol_id) const;

  void append_models_from_directory(
    std::list<std::string> &file_list,
    const std::string &dir_path);

  bool is_imported_module(const std::string &module_name);

  contextt &context;
  namespacet ns;
  typet current_element_type;
  std::string main_python_file;
  std::string current_python_file;
  const nlohmann::json &ast_json;
  const global_scope &global_scope_;
  nlohmann::json imported_module_json;
  std::string current_func_name;
  std::string current_class_name;
  exprt *ref_instance;
  bool is_converting_lhs = false;
  bool is_converting_rhs = false;
  bool is_loading_models = false;
  bool is_importing_module = false;
  bool base_ctor_called = false;

  // Map object to list of instance attributes
  std::unordered_map<std::string, std::set<std::string>> instance_attr_map;
  // Map imported modules to their corresponding paths
  std::unordered_map<std::string, std::string> imported_modules;
};
