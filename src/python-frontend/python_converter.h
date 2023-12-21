#pragma once

#include <util/context.h>
#include <nlohmann/json.hpp>

class codet;
class struct_typet;

class python_converter
{
public:
  python_converter(contextt &_context, const nlohmann::json &ast);
  bool convert();

private:
  void get_var_assign(const nlohmann::json &ast_node, codet &target_block);
  void get_compound_assign(const nlohmann::json &ast_node, codet &target_block);
  void
  get_return_statements(const nlohmann::json &ast_node, codet &target_block);
  void get_function_definition(const nlohmann::json &function_node);
  void get_class_definition(const nlohmann::json &class_node);

  locationt get_location_from_decl(const nlohmann::json &ast_node);
  exprt get_expr(const nlohmann::json &element);
  exprt get_unary_operator_expr(const nlohmann::json &element);
  exprt get_binary_operator_expr(const nlohmann::json &element);
  exprt get_logical_operator_expr(const nlohmann::json &element);
  exprt get_conditional_stm(const nlohmann::json &ast_node);
  exprt get_function_call(const nlohmann::json &ast_block);
  exprt get_block(const nlohmann::json &ast_block);

  const nlohmann::json
  find_var_decl(const std::string &var_name, const nlohmann::json &json);
  void adjust_statement_types(exprt &lhs, exprt &rhs) const;
  std::string create_symbol_id() const;
  bool is_constructor_call(const nlohmann::json &json);
  typet get_typet(const std::string &ast_type);
  typet get_typet(const nlohmann::json &elem);
  void get_attributes_from_self(
    const nlohmann::json &method_body,
    struct_typet &clazz);

  contextt &context;
  typet current_element_type;
  std::string python_filename;
  const nlohmann::json &ast_json;
  std::string current_func_name;
  std::string current_class_name;
  exprt *ref_instance;
};
