#include "python_dict_internal.h"

using namespace python_expr;

bool python_dict_handler::handle_subscript_assignment_check(
  python_converter &converter,
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  codet &target_block)
{
  if (target["_type"] != "Subscript")
    return false;

  exprt container_expr = converter.get_expr(target["value"]);
  typet container_type = container_expr.type();

  if (container_expr.is_symbol())
  {
    const symbolt *sym = symbol_table_.find_symbol(container_expr.identifier());
    if (sym)
      container_type = sym->get_type();
  }

  // Use the namespace from converter, not a method that doesn't exist
  namespacet ns(symbol_table_);
  if (container_type.id() == "symbol")
    container_type = ns.follow(container_type);

  if (!is_dict_type(container_type))
    return false;

  // Handle dict[key] = value assignment
  converter.set_converting_rhs(true);
  exprt rhs = converter.get_expr(ast_node["value"]);
  converter.set_converting_rhs(false);

  handle_dict_subscript_assign(
    container_expr,
    get_key_expr(target["slice"]),
    rhs,
    converter.get_location_from_decl(target["slice"]),
    target["slice"],
    target_block);
  return true;
}

bool python_dict_handler::handle_literal_assignment_check(
  python_converter &converter,
  const nlohmann::json &ast_node,
  const exprt &lhs)
{
  if (!ast_node.contains("value") || ast_node["value"].is_null())
    return false;

  if (!is_dict_literal(ast_node["value"]))
    return false;

  create_dict_from_literal(ast_node["value"], lhs);
  converter.set_current_lhs(nullptr);
  return true;
}

bool python_dict_handler::handle_unannotated_literal_check(
  python_converter &converter,
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  const symbol_id &sid)
{
  if (!ast_node.contains("value") || !ast_node["value"].contains("_type"))
    return false;

  if (!is_dict_literal(ast_node["value"]))
    return false;

  locationt location = converter.get_location_from_decl(target);
  std::string module_name = location.get_file().as_string();
  std::string name;

  if (target["_type"] == "Name")
    name = target["id"].get<std::string>();
  else if (target["_type"] == "Attribute")
    name = target["attr"].get<std::string>();

  symbolt symbol = converter.create_symbol(
    module_name, name, sid.to_string(), location, get_dict_struct_type());
  symbol.lvalue = true;
  symbol.file_local = true;
  symbol.is_extern = false;
  symbolt *lhs_symbol = converter.add_symbol_and_get_ptr(symbol);

  exprt lhs = converter.create_lhs_expression(target, lhs_symbol, location);
  create_dict_from_literal(ast_node["value"], lhs);
  converter.set_current_lhs(nullptr);
  return true;
}
