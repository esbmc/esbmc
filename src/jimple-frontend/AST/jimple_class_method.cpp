#include <jimple-frontend/AST/jimple_class_member.h>
#include <util/std_code.h>
#include <util/expr_util.h>
#include <util/message/format.h>

exprt jimple_class_method::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &file_name) const
{
  exprt dummy;
  code_typet method_type;
  typet inner_type;
  inner_type = t.to_typet();
  method_type.return_type() = inner_type;

  auto id = get_method_name(class_name, name);
  auto symbol = create_jimple_symbolt(method_type, class_name, name, id);

  std::string symbol_name = symbol.id.as_string();

  symbol.lvalue = true;
  symbol.is_extern = false;
  symbol.file_local = true;

  ctx.move_symbol_to_context(symbol);
  symbolt &added_symbol = *ctx.find_symbol(symbol_name);

  // TODO: check for static
  if(1)
  {
    auto this_type = int_type();
    std::string param_id, param_name;

    std::ostringstream oss;
    oss << "@this";
    auto temp = get_symbol_name(class_name, name, oss.str());
    param_id = temp;
    param_name = oss.str();

    auto param_symbol =
      create_jimple_symbolt(this_type, class_name, param_name, param_id, id);
    param_symbol.lvalue = true;
    param_symbol.is_parameter = true;
    param_symbol.file_local = true;

    ctx.move_symbol_to_context(param_symbol);
  }
  for(auto i = 0; i < parameters.size(); i++)
  {
    auto param_type = parameters[i]->to_typet();

    code_typet::argumentt param;

    std::string param_id, param_name;

    std::ostringstream oss;
    oss << "@parameter" << i;
    auto temp = get_symbol_name(class_name, name, oss.str());
    param_id = temp;
    param_name = oss.str();
    param = code_typet::argumentt();
    param.type() = param_type;
    param.cmt_base_name(param_name);
    param.cmt_identifier(param_id);

    auto param_symbol =
      create_jimple_symbolt(param_type, class_name, param_name, param_id, id);
    param_symbol.lvalue = true;
    param_symbol.is_parameter = true;
    param_symbol.file_local = true;

    ctx.move_symbol_to_context(param_symbol);
    method_type.arguments().push_back(param);
  }

  // Apparently, if the type has no arguments, we assume ellipsis
  if(!method_type.arguments().size())
    method_type.make_ellipsis();

  added_symbol.type = method_type;
  added_symbol.value = body->to_exprt(ctx, class_name, this->name);

  return dummy;
}

void jimple_class_method::from_json(const json &j)
{
  // Method modifiers
  auto modifiers = j.at("modifiers");
  m = modifiers.get<jimple_modifiers>();

  // Method type
  j.at("type").get_to(t);

  for(auto x : j.at("parameters"))
  {
    jimple_type t;
    x.get_to(t);
    parameters.push_back(std::make_shared<jimple_type>(t));
  }

  // Method Name
  j.at("name").get_to(this->name);
  name += "_" + get_hash_name();
  try
  {
    j.at("throws").get_to(this->throws);
  }
  catch(std::exception &e)
  {
    this->throws = "(No throw)";
  }

  // TODO: Empty body
  auto j_body = j.at("content");
  // this is a little hacky...
  auto values = j_body.get<jimple_full_method_body>();
  this->body = std::make_shared<jimple_full_method_body>(values);
}
std::string jimple_class_method::to_string() const
{
  std::ostringstream oss;
  oss << "Class Method"
      << "\n\tName: " << this->name << "\n\t" << this->t.to_string() << "\n\t"
      << this->m.to_string() << "\n\tParameters: "
      << "[]" //TODO: this->parameters
      << "\n\tThrows: " << this->throws
      << "\n\tBody : " << this->body->to_string();

  return oss.str();
}
