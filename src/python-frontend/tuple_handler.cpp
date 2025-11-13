#include <python-frontend/tuple_handler.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/symbol_id.h>
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
  std::string tag_name = "tag-tuple";
  for (const auto &type : element_types)
  {
    tag_name += "_" + type.to_string();
  }
  return tag_name;
}

struct_typet tuple_handler::create_tuple_struct_type(
  const std::vector<typet> &element_types) const
{
  struct_typet tuple_type;

  // Add components for each element
  for (size_t i = 0; i < element_types.size(); i++)
  {
    std::string comp_name = "element_" + std::to_string(i);
    struct_typet::componentt comp(comp_name, comp_name, element_types[i]);
    tuple_type.components().push_back(comp);
  }

  // Set the tag to ensure type identity
  tuple_type.tag(build_tuple_tag(element_types));

  return tuple_type;
}

exprt tuple_handler::get_tuple_expr(const nlohmann::json &element)
{
  assert(element.contains("_type") && element["_type"] == "Tuple");
  assert(element.contains("elts"));

  const nlohmann::json &elts = element["elts"];

  // Process each element and collect expressions
  std::vector<exprt> element_exprs;
  std::vector<typet> element_types;
  element_exprs.reserve(elts.size());
  element_types.reserve(elts.size());

  // First pass: get all expressions to determine types
  for (size_t i = 0; i < elts.size(); i++)
  {
    exprt elem_expr = converter_.get_expr(elts[i]);
    element_exprs.push_back(elem_expr);
    element_types.push_back(elem_expr.type());
  }

  // Create struct type for the tuple
  struct_typet tuple_type = create_tuple_struct_type(element_types);

  // Create struct expression with tuple type
  struct_exprt tuple_expr(tuple_type);
  tuple_expr.operands() = element_exprs;

  // Set location information
  if (element.contains("lineno"))
  {
    locationt loc = converter_.get_location_from_decl(element);
    tuple_expr.location() = loc;
  }

  return tuple_expr;
}

bool tuple_handler::is_tuple_type(const typet &type) const
{
  if (!type.is_struct())
    return false;

  const struct_typet &struct_type = to_struct_type(type);
  return struct_type.tag().as_string().find("tag-tuple") == 0;
}

exprt tuple_handler::handle_tuple_subscript(
  const exprt &array,
  const nlohmann::json &slice,
  const nlohmann::json &element)
{
  assert(array.type().is_struct());
  const struct_typet &tuple_type = to_struct_type(array.type());

  // Verify it's a tuple
  if (!is_tuple_type(tuple_type))
  {
    throw std::runtime_error(
      "Subscript on non-tuple struct type: " + tuple_type.tag().as_string());
  }

  // Convert subscript to member access
  exprt index_expr = converter_.get_expr(slice);

  // Index must be a constant for struct member access
  if (!index_expr.is_constant())
  {
    throw std::runtime_error(
      "Tuple subscript with non-constant index is not supported");
  }

  const constant_exprt &const_index = to_constant_expr(index_expr);
  BigInt index_val = binary2integer(const_index.value().c_str(), false);

  // Convert BigInt to size_t for array indexing
  size_t idx = index_val.to_int64();

  // Check bounds
  if (index_val < 0 || idx >= tuple_type.components().size())
  {
    throw std::runtime_error(
      "Tuple index " + integer2string(index_val) + " out of range (size: " +
      std::to_string(tuple_type.components().size()) + ")");
  }

  // Create member access expression: t[0] -> t.element_0
  std::string member_name = "element_" + integer2string(index_val);
  const struct_typet::componentt &comp = tuple_type.components()[idx];

  exprt result = member_exprt(array, member_name, comp.type());

  if (element.contains("lineno"))
  {
    locationt loc = converter_.get_location_from_decl(element);
    result.location() = loc;
  }

  return result;
}

exprt tuple_handler::prepare_rhs_for_unpacking(
  const nlohmann::json &ast_node,
  exprt &rhs,
  codet &target_block)
{
  // If RHS is a function call, we need to create a temporary variable first
  // because we can't do member access on a side effect expression
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
  const struct_typet &tuple_type = to_struct_type(rhs.type());

  // Verify it's a tuple
  if (!is_tuple_type(tuple_type))
  {
    throw std::runtime_error(
      "Cannot unpack non-tuple type: " + tuple_type.tag().as_string());
  }

  const auto &targets = target["elts"];

  if (targets.size() != tuple_type.components().size())
  {
    throw std::runtime_error(
      "Cannot unpack tuple: expected " + std::to_string(targets.size()) +
      " values, got " + std::to_string(tuple_type.components().size()));
  }

  // Create assignments: x = temp.element_0, y = temp.element_1, ...
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

    if (!var_symbol)
    {
      locationt loc = converter_.get_location_from_decl(targets[i]);
      const typet &elem_type = tuple_type.components()[i].type();

      symbolt new_symbol = converter_.create_symbol(
        loc.get_file().as_string(),
        var_name,
        var_sid.to_string(),
        loc,
        elem_type);
      new_symbol.lvalue = true;
      new_symbol.file_local = true;
      new_symbol.is_extern = false;
      var_symbol = converter_.symbol_table().move_symbol_to_context(new_symbol);
    }

    // Create member access: temp.element_i
    std::string member_name = "element_" + std::to_string(i);
    member_exprt member_access(
      rhs, member_name, tuple_type.components()[i].type());

    // Create assignment
    code_assignt assign(symbol_expr(*var_symbol), member_access);
    assign.location() = converter_.get_location_from_decl(ast_node);
    target_block.copy_to_operands(assign);
  }
}

typet tuple_handler::get_tuple_type_from_annotation(
  const nlohmann::json &annotation_node)
{
  assert(
    annotation_node["_type"] == "Subscript" &&
    annotation_node["value"]["id"] == "tuple");

  struct_typet tuple_type;
  const auto &slice = annotation_node["slice"];

  // Build tag name matching the pattern used in get_tuple_expr
  std::vector<typet> element_types;

  if (slice.contains("elts"))
  {
    // Multiple element types: tuple[int, str, float]
    const auto &elts = slice["elts"];
    for (size_t i = 0; i < elts.size(); i++)
    {
      typet elem_type;
      if (elts[i].contains("id"))
        elem_type = type_handler_.get_typet(elts[i]["id"].get<std::string>());
      else
        elem_type = type_handler_.get_typet(elts[i]);

      element_types.push_back(elem_type);
    }
  }

  return create_tuple_struct_type(element_types);
}
