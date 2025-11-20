#include "python_dict_handler.h"
#include <python-frontend/python_converter.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/char_utils.h>
#include <util/c_types.h>
#include <util/arith_tools.h>
#include <util/context.h>
#include <util/python_types.h>
#include <sstream>

int python_dict_handler::dict_counter_ = 0;

python_dict_handler::python_dict_handler(
  python_converter &converter,
  contextt &symbol_table,
  type_handler &type_handler)
  : converter_(converter),
    symbol_table_(symbol_table),
    type_handler_(type_handler)
{
}

bool python_dict_handler::is_dict_literal(const nlohmann::json &element) const
{
  return element.contains("_type") && element["_type"] == "Dict";
}

std::string
python_dict_handler::extract_dict_key(const nlohmann::json &key_node) const
{
  if (key_node["_type"] == "Constant" && key_node["value"].is_string())
    return key_node["value"].get<std::string>();

  if (key_node["_type"] == "Name")
    return key_node["id"].get<std::string>();

  throw std::runtime_error(
    "Dictionary keys must be string literals or identifiers");
}

typet python_dict_handler::infer_value_type(const nlohmann::json &value_node)
{
  if (value_node["_type"] == "Constant")
  {
    const auto &val = value_node["value"];

    if (val.is_string())
      return gen_pointer_type(char_type());

    if (val.is_number_integer())
      return long_long_int_type();

    if (val.is_number_float())
      return double_type();

    if (val.is_boolean())
      return bool_type();

    if (val.is_null())
      return any_type();
  }

  if (value_node["_type"] == "Name")
  {
    // For Name references in dict values, we default to any_type
    // This allows the value to be assigned later with proper typing
    return any_type();
  }

  if (value_node["_type"] == "List")
    return type_handler_.get_list_type();

  if (value_node["_type"] == "Dict")
  {
    // Nested dictionary - create a nested struct type
    return create_dict_struct_type(value_node, "nested_dict");
  }

  // Default to void* for unknown types
  return any_type();
}

struct_typet python_dict_handler::create_dict_struct_type(
  const nlohmann::json &dict_node,
  const std::string &dict_name)
{
  (void)dict_name; // Unused parameter, but kept for API consistency

  // Generate unique struct name
  std::ostringstream struct_name;
  struct_name << "tag-dict_" << dict_counter_++;

  struct_typet dict_struct;
  dict_struct.tag(struct_name.str());

  const auto &keys = dict_node["keys"];
  const auto &values = dict_node["values"];

  if (keys.size() != values.size())
  {
    throw std::runtime_error("Dictionary keys and values size mismatch");
  }

  // Create a component for each key-value pair
  for (size_t i = 0; i < keys.size(); ++i)
  {
    std::string key = extract_dict_key(keys[i]);
    typet value_type = infer_value_type(values[i]);

    struct_typet::componentt component(key, key, value_type);
    component.set_access("public");
    dict_struct.components().push_back(component);
  }

  return dict_struct;
}

exprt python_dict_handler::get_dict_literal(const nlohmann::json &element)
{
  if (!is_dict_literal(element))
  {
    throw std::runtime_error("Expected Dict literal");
  }

  // Extract variable name if this is part of an assignment
  std::string dict_name = "dict_literal";

  // Create struct type for this dictionary
  struct_typet dict_type = create_dict_struct_type(element, dict_name);

  // Register the struct type in the symbol table
  symbolt type_symbol;
  type_symbol.id = dict_type.tag().as_string();
  type_symbol.name = dict_type.tag().as_string();
  type_symbol.type = dict_type;
  type_symbol.mode = "Python";
  type_symbol.is_type = true;
  symbol_table_.add(type_symbol);

  // Create struct expression and initialize fields
  struct_exprt dict_expr(dict_type);

  const auto &values = element["values"];

  for (size_t i = 0; i < values.size(); ++i)
  {
    exprt value_expr = converter_.get_expr(values[i]);

    // Handle string values - convert arrays to pointers
    if (
      value_expr.type().is_array() &&
      dict_type.components()[i].type().is_pointer())
    {
      value_expr =
        converter_.get_string_handler().get_array_base_address(value_expr);
    }

    dict_expr.operands().push_back(value_expr);
  }

  return dict_expr;
}

exprt python_dict_handler::handle_dict_subscript(
  const exprt &dict_expr,
  const nlohmann::json &slice_node)
{
  // Extract the key being accessed
  if (slice_node["_type"] != "Constant" && slice_node["_type"] != "Name")
  {
    throw std::runtime_error(
      "Dictionary subscript must be a string literal or identifier");
  }

  std::string key = extract_dict_key(slice_node);

  // Get the dictionary type
  typet dict_type = dict_expr.type();

  // If dict_expr is a symbol, resolve its type
  if (dict_expr.is_symbol())
  {
    const symbolt *sym = symbol_table_.find_symbol(dict_expr.identifier());
    if (sym)
      dict_type = sym->type;
  }

  // Follow type symbols
  if (dict_type.id() == "symbol")
  {
    namespacet ns(symbol_table_);
    dict_type = ns.follow(dict_type);
  }

  if (!dict_type.is_struct())
  {
    throw std::runtime_error(
      "Dictionary subscript requires a struct type, got: " +
      dict_type.id_string());
  }

  const struct_typet &struct_type = to_struct_type(dict_type);

  // Find the component with matching name
  bool found = false;
  typet field_type;

  for (const auto &component : struct_type.components())
  {
    if (component.get_name() == key)
    {
      field_type = component.type();
      found = true;
      break;
    }
  }

  if (!found)
  {
    throw std::runtime_error("KeyError: '" + key + "' not found in dictionary");
  }

  // Create member access expression
  member_exprt member_expr(dict_expr, key, field_type);

  // Handle pointer dereferencing if needed
  exprt &base = member_expr.struct_op();
  if (base.type().is_pointer())
  {
    exprt deref("dereference");
    deref.type() = base.type().subtype();
    deref.move_to_operands(base);
    base.swap(deref);
  }

  return member_expr;
}
