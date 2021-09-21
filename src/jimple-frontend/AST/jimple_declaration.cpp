//
// Created by rafaelsamenezes on 21/09/2021.
//

#include <jimple-frontend/AST/jimple_declaration.h>

std::string jimple_declaration::to_string()
{
  std::ostringstream oss;
  oss << "Declaration: ";
  for(auto &x : this->names)
    oss << " " << x;
  return oss.str();
}
void jimple_declaration::from_json(const json &j)
{
  j.at("names").get_to(this->names);
  j.at("names").get_to(this->names);
}
