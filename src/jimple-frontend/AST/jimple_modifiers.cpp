#include <jimple-frontend/AST/jimple_modifiers.h>
void jimple_modifiers::from_json(const json &j)
{
  auto modifiers_list = j.get<std::vector<std::string>>();
  for (auto &modifier : modifiers_list)
  {
    this->modifiers.push_back(from_string(modifier));
  }
}

jimple_modifiers::modifier
jimple_modifiers::from_string(const std::string &name) const
{
  return from_map.at(name);
}
std::string
jimple_modifiers::to_string(const jimple_modifiers::modifier &ft) const
{
  return to_map.at(ft);
}
std::string jimple_modifiers::to_string() const
{
  std::ostringstream oss;

  oss << "Modifiers: ||";
  for (auto &modifier : this->modifiers)
  {
    oss << " " << to_string(modifier) << " |";
  }
  oss << "|";

  return oss.str();
}
