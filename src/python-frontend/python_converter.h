#pragma once

#include "util/context.h"
#include <nlohmann/json.hpp>

class python_converter
{
public:
  python_converter(contextt &_context, const std::string &_ast_output_dir)
    : context(_context), ast_output_dir(_ast_output_dir)
  {
  }
  bool convert();

private:
  symbolt get_var_decl(const nlohmann::json &ast_node);
  exprt get_expr(const nlohmann::json &element);
  exprt get_unary_operator_expr(const nlohmann::json &element);
  exprt get_binary_operator_expr(const nlohmann::json &element);

  contextt &context;
  std::string ast_output_dir;
  typet current_element_type;
  nlohmann::json ast_json;
};
