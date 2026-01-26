#include <python-frontend/tuple_handler.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/python_list.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/std_code.h>
#include <util/std_expr.h>

tuple_handler::tuple_handler(
  python_converter &converter,
  type_handler &type_handler)
  : converter_(converter), type_handler_(type_handler)
{
}

std::string
tuple_handler::build_tuple_tag(const std::vector<typet> &element_types) const
{
  // Not used in list-based implementation
  return "tag-tuple";
}

struct_typet tuple_handler::create_tuple_struct_type(
  const std::vector<typet> &element_types) const
{
  // Not used in list-based implementation
  struct_typet tuple_type;
  tuple_type.tag("tag-tuple");
  return tuple_type;
}

exprt tuple_handler::get_tuple_expr(const nlohmann::json &element)
{
  assert(element.contains("_type") && element["_type"] == "Tuple");
  assert(element.contains("elts"));

  // Represent tuple as a list internally
  // Create a modified JSON node that looks like a List instead of Tuple
  nlohmann::json list_node = element;
  list_node["_type"] = "List";

  // Use python_list to create the actual list expression
  python_list tuple_as_list(converter_, list_node);
  return tuple_as_list.get();
}

bool tuple_handler::is_tuple_type(const typet &type) const
{
  // Tuples are represented as lists, so check for list type
  if (!type.is_struct())
    return false;

  const struct_typet &struct_type = to_struct_type(type);
  std::string tag = struct_type.tag().as_string();

  // Accept both tuple and list types since they're now unified
  return tag.find("tag-dynamic_list") == 0 || tag.find("tag-tuple") == 0;
}

exprt tuple_handler::handle_tuple_subscript(
  const exprt &array,
  const nlohmann::json &slice,
  const nlohmann::json &element)
{
  // Delegate to python_list's index method
  python_list tuple_as_list(converter_, element);
  return tuple_as_list.index(array, slice);
}

exprt tuple_handler::prepare_rhs_for_unpacking(
  const nlohmann::json &ast_node,
  exprt &rhs,
  codet &target_block)
{
  if (rhs.is_function_call() || rhs.id() == "sideeffect")
  {
    locationt loc = converter_.get_location_from_decl(ast_node);
    std::string temp_name =
      "ESBMC_unpack_temp_" +
      std::to_string(reinterpret_cast<uintptr_t>(&ast_node));

    symbol_id temp_sid = converter_.create_symbol_id();
    temp_sid.set_object(temp_name);

    symbolt temp_symbol = converter_.create_symbol(
      loc.get_file().as_string(),
      temp_name,
      temp_sid.to_string(),
      loc,
      rhs.type());
    temp_symbol.lvalue = true;
    temp_symbol.file_local = true;
    temp_symbol.is_extern = false;
    temp_symbol.static_lifetime = false;

    symbolt *added_temp =
      converter_.symbol_table().move_symbol_to_context(temp_symbol);
    exprt temp_var = symbol_expr(*added_temp);

    if (rhs.is_function_call())
    {
      code_function_callt &call = static_cast<code_function_callt &>(rhs);
      call.lhs() = temp_var;
      target_block.copy_to_operands(rhs);
    }
    else
    {
      code_assignt temp_assign(temp_var, rhs);
      temp_assign.location() = loc;
      target_block.copy_to_operands(temp_assign);
    }

    return temp_var;
  }

  return rhs;
}

void tuple_handler::handle_tuple_unpacking(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  exprt &rhs,
  codet &target_block)
{
  const auto &targets = target["elts"];

  // Use list indexing to unpack tuple elements
  python_list tuple_as_list(converter_, ast_node);

  // Create assignments: x = tuple[0], y = tuple[1], ...
  for (size_t i = 0; i < targets.size(); i++)
  {
    if (targets[i]["_type"] != "Name")
    {
      throw std::runtime_error(
        "Tuple unpacking only supports simple names, not " +
        targets[i]["_type"].get<std::string>());
    }

    std::string var_name = targets[i]["id"].get<std::string>();
    symbol_id var_sid = converter_.create_symbol_id();
    var_sid.set_object(var_name);

    symbolt *var_symbol = converter_.find_symbol(var_sid.to_string());

    // Create index expression for tuple[i]
    nlohmann::json index_node;
    index_node["_type"] = "Constant";
    index_node["value"] = i;

    exprt indexed_value = tuple_as_list.index(rhs, index_node);

    if (!var_symbol)
    {
      locationt loc = converter_.get_location_from_decl(targets[i]);

      symbolt new_symbol = converter_.create_symbol(
        loc.get_file().as_string(),
        var_name,
        var_sid.to_string(),
        loc,
        indexed_value.type());
      new_symbol.lvalue = true;
      new_symbol.file_local = true;
      new_symbol.is_extern = false;
      var_symbol = converter_.symbol_table().move_symbol_to_context(new_symbol);
    }

    // Create assignment
    code_assignt assign(symbol_expr(*var_symbol), indexed_value);
    assign.location() = converter_.get_location_from_decl(ast_node);
    target_block.copy_to_operands(assign);
  }
}

typet tuple_handler::get_tuple_type_from_annotation(
  const nlohmann::json &annotation_node)
{
  // Return list type since tuples are represented as lists
  // The actual type will be determined dynamically when the tuple is created
  struct_typet list_type;
  list_type.tag("tag-dynamic_list");
  return list_type;
}