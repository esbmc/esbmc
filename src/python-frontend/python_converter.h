#pragma once

#include <util/context.h>
#include <nlohmann/json.hpp>

class codet;

class python_converter
{
public:
  python_converter(contextt &_context, const nlohmann::json &ast)
    : context(_context), ast_json(ast), current_func_name("")
  {
  }
  bool convert();

private:
  void get_var_assign(const nlohmann::json &ast_node, codet &target_block);
  void get_compound_assign(const nlohmann::json &ast_node, codet &target_block);
  void
  get_return_statements(const nlohmann::json &ast_node, codet &target_block);
  void get_function_definition(const nlohmann::json &function_node);

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

  contextt &context;
  typet current_element_type;
  std::string python_filename;
  const nlohmann::json &ast_json;
  std::string current_func_name;
};
