#pragma once

#include "util/context.h"
#include <nlohmann/json.hpp>

class codet;

class python_converter
{
public:
  python_converter(contextt &_context, const nlohmann::json &ast)
    : context(_context), ast_json(ast)
  {
  }
  bool convert();

private:
  void get_var_assign(const nlohmann::json &ast_node, codet &target_block);
  void get_if_statement(const nlohmann::json &ast_node, codet &target_block);
  exprt get_expr(const nlohmann::json &element);
  exprt get_unary_operator_expr(const nlohmann::json &element);
  exprt get_binary_operator_expr(const nlohmann::json &element);
  exprt get_block(const nlohmann::json &ast_block);

  const nlohmann::json find_var_decl(const std::string &id);

  contextt &context;
  typet current_element_type;
  std::string current_function_name;
  const nlohmann::json &ast_json;
};
