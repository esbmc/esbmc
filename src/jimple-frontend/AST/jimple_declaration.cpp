#include <jimple-frontend/AST/jimple_declaration.h>

exprt jimple_declaration::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  typet t = this->type.to_typet(ctx);

  std::string id, name;
  id = get_symbol_name(class_name, function_name, this->name);
  name = this->name;

  auto symbol = create_jimple_symbolt(t, class_name, name, id, function_name);

  symbol.lvalue = true;
  symbol.static_lifetime = false;
  symbol.is_extern = false;
  symbol.file_local = true;

  symbolt &added_symbol = *ctx.move_symbol_to_context(symbol);
  code_declt decl(symbol_expr(added_symbol));
  decl.location() = get_location(class_name, function_name);
  return decl;
}

std::string jimple_declaration::to_string() const
{
  std::ostringstream oss;
  oss << "Declaration: ";
  oss << " " << this->name;
  oss << " | " << type.to_string();

  return oss.str();
}
void jimple_declaration::from_json(const json &j)
{
  j.at("name").get_to(this->name);
  j.at("type").get_to(this->type);
}
