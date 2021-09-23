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
std::string jimple_identity::to_string() const
{
  std::ostringstream oss;
  oss << "Identity: @" << at_identifier
      << "." << this->local_name << " | "
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
