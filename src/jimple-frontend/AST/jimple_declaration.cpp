//
// Created by rafaelsamenezes on 21/09/2021.
//

#include <jimple-frontend/AST/jimple_declaration.h>

std::string jimple_declaration::to_string() const
{
  std::ostringstream oss;
  oss << "Declaration: ";
  for(auto &x : this->names)
    oss << " " << x;
  oss << " | " << t.to_string();

  return oss.str();
}
void jimple_declaration::from_json(const json &j)
{
  j.at("names").get_to(this->names);
  j.at("type").get_to(this->t);
}
