//
// Created by rafaelsamenezes on 22/09/2021.
//

#include <jimple-frontend/AST/jimple_statement.h>
void jimple_identity::from_json(const json &j)
{
  j.at("identifier").get_to(at_identifier);
  j.at("name").get_to(local_name);
  j.at("type").get_to(t);
}

exprt jimple_identity::to_exprt(contextt &ctx, const std::string &class_name, const std::string &function_name) const {

  // TODO: Symbol-table / Typecast
  exprt val("at_identifier");
  symbolt &added_symbol = *ctx.find_symbol(local_name);
  symbolt rhs;
  rhs.name = "@" + at_identifier;
  rhs.id = "@" + at_identifier;
  code_assignt assign(symbol_expr(added_symbol), symbol_expr(rhs));
  return assign;

}
std::string jimple_identity::to_string() const
{
  std::ostringstream oss;
  oss << "Identity:  " << this->local_name
      << " = @"<< at_identifier
      <<  " | "
      << t.to_string();
  return oss.str();
}
std::string jimple_invoke::to_string() const
{
  return "Invoke: (Not implemented)";
}
void jimple_invoke::from_json(const json &j)
{
  // TODO
}

exprt jimple_return::to_exprt(contextt &ctx, const std::string &class_name, const std::string &function_name) const
{
  // TODO: jimple return with support to other returns
  typet return_type = empty_typet();
  code_returnt ret_expr;
  // TODO: jimple return should support values
  return ret_expr;
}

std::string jimple_return::to_string() const
{
  return "Return: (Nothing)";
}
void jimple_return::from_json(const json &j)
{
  // TODO
}
std::string jimple_label::to_string() const
{
  std::ostringstream oss;
  oss << "Label: " << this->label;
  return oss.str();
}

exprt jimple_label::to_exprt(contextt &ctx, const std::string &class_name, const std::string &function_name) const
{
  code_labelt label;
  label.set_label(this->label);
  return label;
}

void jimple_label::from_json(const json &j)
{
  j.get_to(label);
}
std::string jimple_assignment::to_string() const
{
  std::ostringstream oss;
  oss << "Assignment: " << variable
      << " = (Not implemented)";
  return oss.str();
}

void jimple_assignment::from_json(const json &j)
{
  j.at("name").get_to(variable);
}
