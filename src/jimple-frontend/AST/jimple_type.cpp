//
// Created by rafaelsamenezes on 21/09/2021.
//

#include <jimple-frontend/AST/jimple_type.h>

void jimple_type::from_json(const json &j)
{
  // Non-void type
  // TODO: j.at_to("mode")
  j.at("identifier").get_to(name);
  j.at("dimensions").get_to(dimensions);

}
std::string jimple_type::to_string() const
{
  std::ostringstream oss;
  oss << "Type: " << name
      << " [" << dimensions << "]";
  return oss.str();
}
