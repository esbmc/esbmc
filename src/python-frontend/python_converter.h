#pragma once

#include "util/context.h"
#include <nlohmann/json.hpp>

class python_converter
{
public:
  python_converter(contextt &_context) : context(_context)
  {
  }
  bool convert();

private:
  symbolt get_var_decl(const nlohmann::json &ast_node);
  exprt get_expr(const nlohmann::json &element);
  exprt get_unary_operator_expr(const nlohmann::json &element);
  exprt get_binary_operator_expr(const nlohmann::json &element);

  contextt &context;
  typet current_element_type;
};
