//
// Created by rafaelsamenezes on 21/09/2021.
//

#include <jimple-frontend/AST/jimple_type.h>

void jimple_void_type::from_json(const json &j)
{
  // This doesn't need a parse, just an assertion
  auto type = j.get<std::string>();
  assert(type == "void");
}
std::string jimple_void_type::to_string()
{
  return "Type: void";
}
