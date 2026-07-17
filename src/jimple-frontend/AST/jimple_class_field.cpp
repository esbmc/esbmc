#include <jimple-frontend/AST/jimple_class_member.h>
#include <util/std_code.h>
#include <util/expr_util.h>

exprt jimple_class_field::to_exprt(
  contextt &ctx,
  const std::string &,
  const std::string &) const
{
  typet t = type.to_typet(ctx);
  std::string id;
  id = "tag-" + name;
  struct_union_typet::componentt comp(id, name, t);
  return comp;
}

void jimple_class_field::from_json(const json &j)
{
  modifiers = j.at("modifiers").get<jimple_modifiers>();
  j.at("type").get_to(type);
  j.at("name").get_to(this->name);
}
std::string jimple_class_field::to_string() const
{
  std::ostringstream oss;
  oss << "Class Field"
      << "\n\tName: " << this->name << "\n\t" << this->type.to_string()
      << "\n\t" << this->modifiers.to_string();

  return oss.str();
}
