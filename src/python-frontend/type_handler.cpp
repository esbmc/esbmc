#include <python-frontend/type_handler.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/symbol_id.h>
#include <util/context.h>

type_handler::type_handler(const python_converter &converter)
  : converter_(converter)
{
}

bool type_handler::is_constructor_call(const nlohmann::json &json) const
{
  if (
    !json.contains("_type") || json["_type"] != "Call" ||
    !json["func"].contains("id"))
    return false;

  const std::string &func_name = json["func"]["id"];

  if (type_utils::is_builtin_type(func_name))
    return false;

  /* The statement is a constructor call if the function call on the
   * rhs corresponds to the name of a class. */

  bool is_ctor_call = false;

  const contextt &symbol_table = converter_.symbol_table();

  symbol_table.foreach_operand([&](const symbolt &s) {
    if (s.type.id() == "struct" && s.name == func_name)
    {
      is_ctor_call = true;
      return;
    }
  });

  return is_ctor_call;
}

std::string type_handler::type_to_string(const typet &t) const
{
  if (t == double_type())
    return "float";
  if (t == long_long_int_type())
    return "int";
  if (t == long_long_uint_type())
    return "uint64";
  if (t == bool_type())
    return "bool";
  if (t == uint256_type())
    return "uint256";
  if (t.is_array())
  {
    const array_typet &arr_type = static_cast<const array_typet &>(t);
    if (arr_type.subtype() == char_type())
      return "str";
    if (arr_type.subtype() == int_type())
      return "bytes";
    if (arr_type.subtype().is_array())
      return type_to_string(arr_type.subtype());
  }

  return "";
}

std::string type_handler::get_var_type(const std::string &var_name) const
{
  nlohmann::json ref = json_utils::find_var_decl(
    var_name, converter_.current_function_name(), converter_.ast());

  if (ref.empty())
    return std::string();

  return ref["annotation"]["id"].get<std::string>();
}

typet type_handler::build_array(const typet &sub_type, const size_t size) const
{
  return array_typet(
    sub_type,
    constant_exprt(
      integer2binary(BigInt(size), bv_width(size_type())),
      integer2string(BigInt(size)),
      size_type()));
}

// Convert Python/AST types to irep types
typet type_handler::get_typet(const std::string &ast_type, size_t type_size)
  const
{
  if (ast_type == "float")
    return double_type();
  if (ast_type == "int" || ast_type == "GeneralizedIndex")
    /* FIXME: We need to map 'int' to another irep type that provides unlimited precision
  	https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex */
    return long_long_int_type();
  if (
    ast_type == "uint" || ast_type == "uint64" || ast_type == "Epoch" ||
    ast_type == "Slot")
    return long_long_uint_type();
  if (ast_type == "bool")
    return bool_type();
  if (ast_type == "uint256" || ast_type == "BLSFieldElement")
    return uint256_type();
  if (ast_type == "bytes")
  {
    // TODO: Keep "bytes" as signed char instead of "int_type()", and cast to an 8-bit integer in [] operations
    // or consider modelling it with string_constantt.
    return build_array(long_long_int_type(), type_size);
  }
  if (ast_type == "str")
  {
    if (type_size == 1)
    {
      typet type = char_type();
      type.set("#cpp_type", "char");
      return type;
    }
    return build_array(char_type(), type_size);
  }
  if (json_utils::is_class(ast_type, converter_.ast()))
    return symbol_typet("tag-" + ast_type);

  return empty_typet();
}

typet type_handler::get_typet(const nlohmann::json &elem) const
{
  if (elem.is_number_integer() || elem.is_number_unsigned())
    return long_long_int_type();
  else if (elem.is_boolean())
    return bool_type();
  else if (elem.is_number_float())
    return float_type();
  else if (elem.is_string())
    return build_array(char_type(), elem.get<std::string>().size());
  else if (elem.is_object() && elem.contains("value"))
    return get_typet(elem["value"]);
  else if (elem.is_array())
  {
    typet subtype = get_typet(elem[0]);
    return build_array(subtype, elem.size());
  }

  throw std::runtime_error("Invalid type");
}

bool type_handler::has_multiple_types(const nlohmann::json &container) const
{
  if (container.empty())
    return false;

  // Determine the type of the first element
  typet t;
  if (container[0]["_type"] == "List")
  {
    // Check the type of elements within the sublist
    if (has_multiple_types(container[0]["elts"]))
      return true;

    // Get the type of the elements in the sublist
    t = get_typet(container[0]["elts"][0]["value"]);
  }
  else
  {
    // Get the type of the first element if it is not a sublist
    t = get_typet(container[0]["value"]);
  }

  for (const auto &element : container)
  {
    if (element["_type"] == "List")
    {
      // Check the consistency of the sublist
      if (has_multiple_types(element["elts"]))
        return true;

      // Compare the type of internal elements in the sublist with the type `t`
      if (get_typet(element["elts"][0]["value"]) != t)
        return true;
    }
    else
    {
      // Compare the type of the current element with `t`
      if (get_typet(element["value"]) != t)
        return true;
    }
  }
  return false;
}

typet type_handler::get_list_type(const nlohmann::json &list_value) const
{
  if (list_value["_type"] == "arg" && list_value.contains("annotation"))
  {
    assert(list_value["annotation"]["value"]["id"] == "list");
    typet t =
      get_typet(list_value["annotation"]["slice"]["id"].get<std::string>());
    return build_array(t, 0);
  }

  if (list_value["_type"] == "List") // Get list value type from elements
  {
    const nlohmann::json &elts = list_value["elts"];

    if (!has_multiple_types(elts)) // All elements have the same type
    {
      typet subtype;

      if (elts[0]["_type"] == "Constant") // One-dimensional list
        // Retrieve the type of the first element
        subtype = get_typet(elts[0]["value"]);
      else // Multi-dimensional list
        // Get sub-array type
        subtype = get_typet(elts[0]["elts"]);

      return build_array(subtype, elts.size());
    }
    throw std::runtime_error("Multiple type lists are not supported yet");
  }

  if (list_value["_type"] == "Call") // Get list type from function return type
  {
    symbol_id sid(
      converter_.python_file(),
      converter_.current_classname(),
      converter_.current_function_name());

    if (list_value["func"]["_type"] == "Attribute")
      sid.set_function(list_value["func"]["attr"]);
    else
      sid.set_function(list_value["func"]["id"]);

    symbolt *func_symbol =
      converter_.symbol_table().find_symbol(sid.to_string());
    if (!func_symbol)
      func_symbol = converter_.find_imported_symbol(sid.to_string());

    assert(func_symbol);
    return static_cast<code_typet &>(func_symbol->type).return_type();
  }

  return typet();
}

std::string type_handler::get_operand_type(const nlohmann::json &operand) const
{
  // Operand is a variable
  if (operand["_type"] == "Name")
    return get_var_type(operand["id"]);

  // Operand is a literal
  if (operand["_type"] == "Constant")
  {
    const auto &value = operand["value"];
    if (value.is_string())
      return "str";
    if (value.is_number_integer() || value.is_number_unsigned())
      return "int";
    else if (value.is_boolean())
      return "bool";
    else if (value.is_number_float())
      return "float";
  }

  // Operand is a list element
  if (
    operand["_type"] == "Subscript" &&
    get_operand_type(operand["value"]) == "list")
  {
    nlohmann::json list_node = json_utils::find_var_decl(
      operand["value"]["id"].get<std::string>(),
      converter_.current_function_name(),
      converter_.ast());

    array_typet list_type = get_list_type(list_node["value"]);
    return type_to_string(list_type.subtype());
  }

  return std::string();
}
