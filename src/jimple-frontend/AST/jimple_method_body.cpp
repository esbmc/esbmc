//
// Created by rafaelsamenezes on 21/09/2021.
//

#include <jimple-frontend/AST/jimple_method_body.h>
#include <jimple-frontend/AST/jimple_declaration.h>
void jimple_full_method_body::from_json(const json &j)
{
  for(auto &x : j)
  {
    if(x.contains("declaration"))
    {
      jimple_declaration d;
      x.at("declaration").get_to(d);
      members.push_back(std::make_shared<jimple_declaration>(d));
    }
  }
}
std::string jimple_full_method_body::to_string()
{
  std::ostringstream oss;
  for(auto &x : members)
  {
    oss << "\n\t\t" << x->to_string();
  }
  return oss.str();
}
