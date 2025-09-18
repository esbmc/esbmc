#include <python-frontend/type_handler.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/symbol_id.h>
#include <util/arith_tools.h>
#include <util/context.h>
#include <util/c_types.h>
#include <util/message.h>

#include <regex>

type_handler::type_handler(const python_converter &converter)
  : converter_(converter)
{
}

exprt type_handler::get_expr_helper(const nlohmann::json &json) const
{
  // This is safe because get_expr doesn't modify the converter's logical state
  return const_cast<python_converter &>(converter_).get_expr(json);
}

bool type_handler::is_constructor_call(const nlohmann::json &json) const
{
  if (
    !json.contains("_type") || json["_type"] != "Call" ||
    (!json["func"].contains("id") && !json["func"].contains("attr")))
    return false;

  const std::string &func_name = json["func"]["_type"] == "Attribute"
                                   ? json["func"]["attr"]
                                   : json["func"]["id"];

  if (func_name == "__init__")
    return true;

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

/// This utility maps internal ESBMC types to their corresponding Python type strings
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
    const array_typet &arr_type = to_array_type(t); // Safer than static_cast

    const typet &elem_type = arr_type.subtype();

    if (elem_type == char_type())
      return "str";

    if (elem_type == int_type())
      return "bytes";

    // Handle nested arrays (e.g., list of strings)
    if (elem_type.is_array())
      return type_to_string(elem_type);
  }

  if (t.is_pointer() && t.subtype() == char_type())
    return "str";

  return "";
}

std::string type_handler::get_var_type(const std::string &var_name) const
{
  nlohmann::json ref = json_utils::find_var_decl(
    var_name, converter_.current_function_name(), converter_.ast());

  if (ref.empty())
    return std::string();

  const auto &annotation = ref["annotation"];
  if (annotation.contains("id"))
    return annotation["id"].get<std::string>();

  if (annotation["_type"] == "Subscript")
    return annotation["value"]["id"];

  return std::string();
}

/// Check if two types are compatible for list homogeneity
/// This considers strings of different lengths as the same type
bool type_handler::are_types_compatible(const typet &t1, const typet &t2) const
{
  // Exact match
  if (t1 == t2)
    return true;

  // Both are character arrays (strings) - consider them compatible
  if (t1.is_array() && t2.is_array())
  {
    const array_typet &arr1 = to_array_type(t1);
    const array_typet &arr2 = to_array_type(t2);

    // If subtypes match, consider them compatible regardless of size
    if (arr1.subtype() == arr2.subtype())
      return true;
  }

  return false;
}

/// Get a normalized/canonical type for list element type inference
/// This ensures all strings use the same representative type regardless of length
typet type_handler::get_canonical_string_type(const typet &t) const
{
  // For string types (char arrays), return a canonical string type
  if (t.is_array())
  {
    const array_typet &arr_type = to_array_type(t);
    if (arr_type.subtype() == char_type())
    {
      // Return a canonical string type (size 0 array indicates variable length string)
      return build_array(char_type(), 0);
    }
  }

  return t;
}

/// This method creates a `typet` representing a statically sized array.
/// It is typically used to model Python sequences like strings and byte arrays
typet type_handler::build_array(const typet &sub_type, const size_t size) const
{
  // Use BigInt to ensure correctness for large sizes, though typical sizes are small.
  const BigInt big_size = BigInt(size);
  const typet size_t_type = size_type(); // An unsignedbv of platform word width

  // Construct a constant expression for the array size.
  constant_exprt array_size_expr(
    integer2binary(big_size, bv_width(size_t_type)), // Binary representation
    integer2string(big_size), // Decimal string for display
    size_t_type);             // Size type

  // Return the full array type
  return array_typet(sub_type, array_size_expr);
}

std::vector<int> type_handler::get_array_type_shape(const typet &type) const
{
  // If the type is not an array, return an empty shape.
  if (!type.is_array())
    return {};

  // Since type is an array, cast it to array_typet.
  const auto &arr_type = static_cast<const array_typet &>(type);
  std::vector<int> shape{
    std::stoi(arr_type.size().value().as_string(), nullptr, 2)};

  // Recursively append dimensions from the subtype.
  auto sub_shape = get_array_type_shape(type.subtype());
  shape.insert(shape.end(), sub_shape.begin(), sub_shape.end());

  return shape;
}

/// Convert a Python AST type to an ESBMC internal irep type.
/// This function maps high-level Python types (from AST) to low-level internal
/// ESBMC representations using `typet`. It supports core built-in types
///
/// References:
/// - Python 3 type system: https://docs.python.org/3/library/stdtypes.html
/// - ESBMC irep type system: src/util/type.h
typet type_handler::get_typet(const std::string &ast_type, size_t type_size)
  const
{
  if (ast_type == "Any")
    return empty_typet();

  // NoneType — represents Python's None value
  // Use a pointer type to void to represent None/null properly
  if (ast_type == "NoneType")
    return pointer_type();

  // Python float type: IEEE 754 double-precision mapping
  // Python floats are implemented using C double (IEEE 754 double-precision)
  // as per Python documentation. This ensures proper precision, range, and
  // compatibility with Python's numeric type promotion (int -> float -> complex).
  if (ast_type == "float")
    return double_type();

  // int — arbitrarily large integers
  // We approximate using 64-bit signed integer here.
  if (ast_type == "int" || ast_type == "GeneralizedIndex")
    return long_long_int_type(); // FIXME: Support bignum for true Python semantics

  // Unsigned integers used in domains like Ethereum or system modeling
  if (
    ast_type == "uint" || ast_type == "uint64" || ast_type == "Epoch" ||
    ast_type == "Slot")
    return long_long_uint_type();

  // bool — represents True/False
  if (ast_type == "bool")
    return bool_type();

  // Custom large unsigned integer types (used in Ethereum, BLS, etc.)
  if (ast_type == "uint256" || ast_type == "BLSFieldElement")
    return uint256_type();

  // bytes — immutable sequences of bytes
  // Here modeled as array of signed integers (8-bit).
  if (ast_type == "bytes")
  {
    // TODO: Refactor to model using unsigned/signed char
    return build_array(long_long_int_type(), type_size);
  }

  // str: immutable sequences of Unicode characters
  // chr(): returns a 1-character string
  // hex(): returns string representation of integer in hex
  // oct(): Converts an integer to a lowercase octal string
  // ord(): Converts a 1-character string to its Unicode code point (as integer)
  // abs(): Return the absolute value of a number
  if (
    ast_type == "str" || ast_type == "chr" || ast_type == "hex" ||
    ast_type == "oct" || ast_type == "ord" || ast_type == "abs")
  {
    if (type_size == 1)
    {
      typet type = char_type();      // 8-bit char
      type.set("#cpp_type", "char"); // For C backend compatibility
      return type;
    }
    return build_array(char_type(), type_size); // Array of characters
  }

  // Custom user-defined types / classes
  if (
    json_utils::is_class(ast_type, converter_.ast()) ||
    type_utils::is_python_exceptions(ast_type))
    return symbol_typet("tag-" + ast_type);

  if (ast_type != "Any")
    log_warning("Unknown or unsupported AST type: {}", ast_type);

  return empty_typet();
}

typet type_handler::get_typet(const nlohmann::json &elem) const
{
  // Handle null/empty values
  if (elem.is_null())
    return empty_typet();

  // Handle primitive types
  if (elem.is_number_integer() || elem.is_number_unsigned())
    return long_long_int_type();
  else if (elem.is_boolean())
    return bool_type();
  else if (elem.is_number_float())
    return double_type();
  else if (elem.is_string())
  {
    size_t str_size = elem.get<std::string>().size();
    if (str_size > 1)
      str_size += 1;
    return build_array(char_type(), str_size);
  }

  // Handle nested value object
  if (elem.is_object())
  {
    // Recursive delegation for wrapper node
    if (elem.contains("value"))
      return get_typet(elem["value"]);

    // Handle Python AST UnaryOp node (e.g., -1, +1, ~1, not x)
    if (elem["_type"] == "UnaryOp" && elem.contains("operand"))
    {
      // For unary operations, the result type is typically the same as the operand type
      return get_typet(elem["operand"]);
    }

    // Handle Python AST List node
    if (elem["_type"] == "List" && elem.contains("elts"))
    {
      const auto &elements = elem["elts"];
      if (elements.empty())
        return build_array(long_long_int_type(), 0);

      typet subtype = get_typet(elements[0]);
      return build_array(subtype, elements.size());
    }
  }

  if (elem.is_array())
  {
    if (elem.empty())
      return build_array(long_long_int_type(), 0);

    typet subtype = get_typet(elem[0]);
    return build_array(subtype, elem.size());
  }

  if (
    elem["_type"] == "Call" && type_utils::is_builtin_type(elem["func"]["id"]))
  {
    return get_typet(elem["func"]["id"].get<std::string>());
  }

  if (elem["_type"] == "Name")
  {
    const nlohmann::json &var = json_utils::find_var_decl(
      elem["id"], converter_.current_function_name(), converter_.ast());

    if (!var.empty() && var.contains("value") && !var["value"].is_null())
    {
      if (var["value"]["_type"] == "Call")
      {
        // Try to handle known patterns before giving up
        if (var["value"].contains("func"))
        {
          const auto &func = var["value"]["func"];

          // Handle simple function calls: func_name()
          if (func.contains("id") && type_utils::is_builtin_type(func["id"]))
            return get_typet(func["id"].get<std::string>());

          // Handle attribute calls: module.func_name()
          if (func["_type"] == "Attribute" && func.contains("attr"))
          {
            std::string attr_name = func["attr"].get<std::string>();
            // Handle common cases like random.randint, math.sqrt, etc.
            if (attr_name == "randint" || attr_name == "randrange")
              return long_long_int_type();
            if (attr_name == "random" || attr_name == "uniform")
              return double_type();
            if (type_utils::is_builtin_type(attr_name))
              return get_typet(attr_name);
          }
        }
        throw std::runtime_error("Invalid type");
      }
      return get_typet(var["value"]["value"]);
    }

    // Fallback for cases where variable declaration has no value or is null
    return empty_typet();
  }

  throw std::runtime_error("Invalid type");
}

bool type_handler::has_multiple_types(const nlohmann::json &container) const
{
  if (container.empty())
    return false;

  // Helper lambda that leverages existing get_typet method
  auto get_element_type = [this](const nlohmann::json &element) -> typet {
    try
    {
      typet elem_type = get_typet(element);
      // For array types, we want the element type, not the container type
      return elem_type.is_array() ? elem_type.subtype() : elem_type;
    }
    catch (const std::exception &)
    {
      log_warning("Failed to determine element type in has_multiple_types");
      return empty_typet();
    }
  };

  // Get canonical type of first element
  typet canonical_first_type =
    get_canonical_string_type(get_element_type(container[0]));

  if (canonical_first_type == empty_typet())
    return false; // Couldn't determine type, assume homogeneous

  // Check all elements for type compatibility
  for (const auto &element : container)
  {
    // Handle nested lists recursively
    if (
      element["_type"] == "List" && element.contains("elts") &&
      !element["elts"].empty())
    {
      if (has_multiple_types(element["elts"]))
        return true;
    }

    // Check type compatibility
    typet element_type = get_canonical_string_type(get_element_type(element));
    if (
      element_type != empty_typet() &&
      !are_types_compatible(canonical_first_type, element_type))
      return true;
  }

  return false;
}

typet type_handler::get_list_type_improved(const nlohmann::json &element)
{
  if (!element.contains("elts") || element["elts"].empty())
    return array_typet(empty_typet(), from_integer(0, size_type()));

  const auto &elements = element["elts"];

  // Check if all elements are string constants
  bool all_strings = true;
  size_t max_string_length = 0;

  for (const auto &elem : elements)
  {
    if (elem["_type"] == "Constant" && elem["value"].is_string())
    {
      std::string str_val = elem["value"].get<std::string>();
      max_string_length = std::max(
        max_string_length, str_val.size() + 1); // +1 for null terminator
    }
    else
    {
      all_strings = false;
      break;
    }
  }

  if (all_strings)
  {
    // Create array of string arrays (char arrays)
    typet string_type = build_array(char_type(), max_string_length);
    return array_typet(string_type, from_integer(elements.size(), size_type()));
  }

  // Fallback to original implementation
  return get_list_type(element);
}

typet type_handler::get_list_type(const nlohmann::json &list_value) const
{
  if (
    list_value.is_null() ||
    (list_value.contains("elts") && list_value["elts"].empty()))
  {
    return build_array(empty_typet(), 0);
  }

  if (list_value["_type"] == "arg" && list_value.contains("annotation"))
  {
    // Handle case where annotation is directly a Subscript (e.g., List['Action'])
    if (list_value["annotation"]["_type"] == "Subscript")
    {
      const nlohmann::json &slice = list_value["annotation"]["slice"];
      typet t;

      if (slice.contains("id"))
      {
        // Regular identifier like List[int]
        t = get_typet(slice["id"].get<std::string>());
      }
      else if (slice["_type"] == "Constant" && slice.contains("value"))
      {
        // String constant like List['Action'] (forward reference)
        std::string type_string = slice["value"].get<std::string>();
        t = get_typet(type_utils::remove_quotes(type_string));
      }
      else
        t = empty_typet();
      return pointer_typet(t);
    }

    // Check if the nested structure exists before accessing
    if (
      list_value["annotation"].contains("value") &&
      list_value["annotation"]["value"].contains("id"))
    {
      const nlohmann::json &type_ann = list_value["annotation"]["value"]["id"];
      assert(type_ann == "list" || type_ann == "List");
      typet t =
        get_typet(list_value["annotation"]["slice"]["id"].get<std::string>());
      return pointer_typet(t);
    }
  }

  if (list_value["_type"] == "List") // Get list value type from elements
  {
    const nlohmann::json &elts = list_value["elts"];

    if (!has_multiple_types(elts)) // All elements have the same type
    {
      typet subtype;

      if (elts[0]["_type"] == "Constant" || elts[0]["_type"] == "UnaryOp")
      { // One-dimensional list
        // Retrieve the type of the first element
        const auto &elem = (elts[0]["_type"] == "UnaryOp")
                             ? elts[0]["operand"]["value"]
                             : elts[0]["value"];
        subtype = get_typet(elem);
      }
      else
      {
        // Get sub-array type from multi-dimensional list
        if (elts[0]["_type"] == "Call")
        {
          if (type_utils::is_builtin_type(elts[0]["func"]["id"]))
            subtype = get_typet(elts[0]["func"]["id"].get<std::string>());
        }
        else if (elts[0].contains("elts"))
          subtype = get_typet(elts[0]["elts"]);
        else
        {
          // Handle other element types directly
          subtype = get_typet(elts[0]);
        }
      }

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

    symbolt *func_symbol = converter_.find_symbol(sid.to_string());

    assert(func_symbol);
    return static_cast<code_typet &>(func_symbol->type).return_type();
  }

  if (list_value.contains("_type") && list_value["_type"] == "BinOp")
  {
    // Handle cases like x = [0] * 5
    if (list_value["op"]["_type"] == "Mult")
    {
      exprt left_expr = get_expr_helper(list_value["left"]);
      exprt right_expr = get_expr_helper(list_value["right"]);

      typet list_type = (left_expr.is_symbol()) ? left_expr.type().subtype()
                                                : right_expr.type().subtype();
      exprt size = (left_expr.is_symbol()) ? right_expr : left_expr;
      return array_typet(list_type, size);
    }
  }

  return typet();
}

const typet type_handler::get_list_type() const
{
  static const symbolt *list_type_symbol = nullptr;
  const char *list_type_id = "tag-struct __anon_typedef_List_at_";

  if (!list_type_symbol)
  {
    converter_.symbol_table().foreach_operand(
      [&list_type_id](const symbolt &s) {
        const std::string &symbol_id = s.id.as_string();
        if (symbol_id.find(list_type_id) != std::string::npos)
        {
          list_type_symbol = &s;
        }
      });
  }

  assert(list_type_symbol);
  return pointer_typet(symbol_typet(list_type_symbol->id));
}

typet type_handler::get_list_element_type() const
{
  static const symbolt *type = nullptr;
  const char *type_id = "tag-struct __anon_typedef_Object_at";

  if (!type)
  {
    converter_.symbol_table().foreach_operand([&type_id](const symbolt &s) {
      const std::string &symbol_id = s.id.as_string();
      if (symbol_id.find(type_id) != symbol_id.npos)
      {
        type = &s;
      }
    });
  }

  assert(type);
  return symbol_typet(type->id);
}

/// This method inspects the JSON representation of a Python operand node and attempts to
/// infer its type based on its AST node type (`_type`). It currently supports variable
/// names, constants (literals), and list subscripts. This type information is used for
/// symbolic execution or translation within ESBMC.
std::string type_handler::get_operand_type(const nlohmann::json &operand) const
{
  // Handle variable reference (e.g., `x`)
  if (
    operand.contains("_type") && operand["_type"] == "Name" &&
    operand.contains("id"))
    return get_var_type(operand["id"]);

  // Handle constant/literal values (e.g., 42, "hello", True, 3.14)
  else if (operand["_type"] == "Constant" && operand.contains("value"))
  {
    const auto &value = operand["value"];
    if (value.is_string())
      return "str";
    else if (value.is_number_integer() || value.is_number_unsigned())
      return "int";
    else if (value.is_boolean())
      return "bool";
    else if (value.is_number_float())
      return "float";
  }

  // Handle list subscript (e.g., `mylist[0]`)
  else if (
    operand["_type"] == "Subscript" && operand.contains("value") &&
    get_operand_type(operand["value"]) == "list")
  {
    const auto &list_expr = operand["value"];
    if (list_expr.contains("id"))
    {
      std::string list_id = list_expr["id"].get<std::string>();

      // Find the declaration of the list variable
      nlohmann::json list_node = json_utils::find_var_decl(
        list_id, converter_.current_function_name(), converter_.ast());

      // Get the type of the list and return the subtype (element type)
      array_typet list_type = get_list_type(list_node["value"]);
      return type_to_string(list_type.subtype());
    }
  }

  // If no known type can be determined, issue a warning and return std::string()
  log_warning(
    "type_handler::get_operand_type: unable to determine operand type for AST "
    "node: {}",
    operand.dump(2));
  return std::string();
}

bool type_handler::is_2d_array(const nlohmann::json &arr) const
{
  return arr.contains("_type") && arr["_type"] == "List" &&
         arr.contains("elts") && !arr["elts"].empty() &&
         arr["elts"][0].is_object() && arr["elts"][0].contains("elts");
}

// Add this method to the type_handler class
int type_handler::get_array_dimensions(const nlohmann::json &arr) const
{
  if (!arr.is_object() || arr["_type"] != "List" || !arr.contains("elts"))
    return 0;

  if (arr["elts"].empty())
    return 1; // Empty array is considered 1D

  // Check the first element to determine nesting depth
  const auto &first_elem = arr["elts"][0];

  if (!first_elem.is_object())
    return 1;

  if (first_elem["_type"] == "List")
  {
    // Recursive case: this is a nested array
    return 1 + get_array_dimensions(first_elem);
  }
  else
  {
    // Base case: first element is not a list, so this is 1D
    return 1;
  }
}

size_t type_handler::get_type_width(const typet &type) const
{
  // First try to parse width directly
  try
  {
    return std::stoi(type.width().c_str());
  }
  catch (const std::exception &)
  {
    // If direct parsing fails, try to infer from type name
    std::string type_str = type.width().as_string();

    // Handle common Python/ESBMC type mappings
    if (type_str == "int32")
      return 32;
    else if (type_str == "int")
      return 64;
    else if (type_str == "int64" || type_str == "long")
      return 64;
    else if (type_str == "int16" || type_str == "short")
      return 16;
    else if (type_str == "int8" || type_str == "char")
      return 8;
    else if (type_str == "float32")
      return 32;
    else if (type_str == "float")
      return 64;
    else if (type_str == "double" || type_str == "float64")
      return 64;
    else if (type_str == "bool")
      return 1;

    // Try to extract number from string like "int32", "uint64", etc.
    std::regex width_regex(R"(\d+)");
    std::smatch match;
    if (std::regex_search(type_str, match, width_regex))
    {
      try
      {
        return std::stoi(match.str());
      }
      catch (const std::exception &)
      {
        // Fall through to default
      }
    }

    // Default to 32 for unknown types
    return 32;
  }
}
