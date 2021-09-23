//
// Created by rafaelsamenezes on 21/09/2021.
//

#include <jimple-frontend/AST/jimple_modifiers.h>
void jimple_modifiers::from_json(const json &j)
{
  auto p = j.get<std::vector<std::string>>();
  for(auto &x : p) {
    this->m.push_back(from_string(x));
  }
}

namespace {
typedef jimple_modifiers::modifier modifier;
std::map<std::string, modifier> from_map = {
  {"Abstract", modifier::Abstract},
  {"Final", modifier::Final},
  {"Native", modifier::Native},
  {"Public", modifier::Public},
  {"Protected", modifier::Protected},
  {"Private", modifier::Private},
  {"Static", modifier::Static},
  {"Synchronized", modifier::Synchronized},
  {"Transient", modifier::Transient},
  {"Volatile", modifier::Volatile},
  {"StrictFp", modifier::StrictFp},
  {"Enum", modifier::Enum},
  {"Annotation", modifier::Annotation}
};

std::map<modifier, std::string> to_map = {
  {modifier::Abstract, "Abstract"},
  {modifier::Final, "Final"},
  {modifier::Native, "Native"},
  {modifier::Public, "Public"},
  {modifier::Protected, "Protected"},
  {modifier::Private, "Private"},
  {modifier::Static, "Static"},
  {modifier::Synchronized, "Synchronized"},
  {modifier::Transient, "Transient"},
  {modifier::Volatile, "Volatile"},
  {modifier::StrictFp, "StrictFp"},
  {modifier::Enum, "Enum"},
  {modifier::Annotation, "Annotation"}
};
}

jimple_modifiers::modifier
jimple_modifiers::from_string(const std::string &name)
{
  return from_map.at(name);
}
std::string jimple_modifiers::to_string(const jimple_modifiers::modifier &ft) const
{
  return to_map.at(ft);
}
std::string jimple_modifiers::to_string() const
{
  std::ostringstream oss;

  oss << "Modifiers: ||";
  for(auto &x : this->m)
  {
    oss << " " << to_string(x) << " |";
  }
  oss << "|";

  return oss.str();
}
