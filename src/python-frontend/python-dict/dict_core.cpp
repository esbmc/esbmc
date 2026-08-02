#include "python_dict_internal.h"

using namespace python_expr;

std::unordered_map<std::string, std::string>
  python_dict_handler::dict_keys_list_id_;
std::unordered_map<std::string, std::string>
  python_dict_handler::dict_vals_list_id_;

python_dict_handler::python_dict_handler(
  python_converter &converter,
  contextt &symbol_table,
  type_handler &type_handler)
  : converter_(converter),
    symbol_table_(symbol_table),
    type_handler_(type_handler)
{
}

std::string python_dict_handler::generate_unique_dict_name(
  const nlohmann::json &element,
  const locationt &location) const
{
  std::ostringstream name;
  name << "$py_dict$";

  // Try to use location information for deterministic naming
  if (!location.get_file().empty() && !location.get_line().empty())
  {
    // Use file name (without path) + line + column
    std::string file = location.get_file().as_string();

    // Extract just the filename without path
    size_t last_slash = file.find_last_of("/\\");
    if (last_slash != std::string::npos)
      file = file.substr(last_slash + 1);

    // Replace dots and special chars with underscores for valid identifiers
    std::replace(file.begin(), file.end(), '.', '_');
    std::replace(file.begin(), file.end(), '-', '_');

    name << file << "$" << location.get_line().as_string() << "$"
         << location.get_column().as_string();
  }
  else
  {
    // Fallback: use hash of the JSON element for uniqueness
    // This handles cases where location info is missing
    std::hash<std::string> hasher;
    size_t hash = hasher(element.dump());
    name << "noloc$" << std::hex << hash;
  }

  // Add a disambiguator based on element content hash to handle
  // multiple dicts at the same location (e.g., in list comprehensions)
  std::hash<std::string> hasher;
  size_t content_hash = hasher(element.dump());
  name << "$" << std::hex << (content_hash & 0xFFFF); // Use last 4 hex digits

  return name.str();
}

bool python_dict_handler::is_dict_literal(const nlohmann::json &element) const
{
  return element.contains("_type") && element["_type"] == "Dict";
}

bool python_dict_handler::is_dict_type(const typet &type) const
{
  if (!type.is_struct())
    return false;

  const struct_typet &struct_type = to_struct_type(type);
  std::string tag = struct_type.tag().as_string();
  return tag == "__python_dict__";
}

struct_typet python_dict_handler::get_dict_struct_type()
{
  const std::string dict_type_name = "tag-__python_dict__";
  symbolt *existing = symbol_table_.find_symbol(dict_type_name);
  if (existing)
    return to_struct_type(existing->get_type());

  struct_typet dict_struct;
  dict_struct.tag("__python_dict__");
  set_python_aggregate_kind(dict_struct, "dict");

  typet list_type = type_handler_.get_list_type();

  struct_typet::componentt keys_comp("keys", "keys", list_type);
  keys_comp.set_access("public");
  dict_struct.components().push_back(keys_comp);

  struct_typet::componentt values_comp("values", "values", list_type);
  values_comp.set_access("public");
  dict_struct.components().push_back(values_comp);

  symbolt type_symbol;
  type_symbol.id = dict_type_name;
  type_symbol.name = dict_type_name;
  type_symbol.set_type(dict_struct);
  type_symbol.mode = "Python";
  type_symbol.is_type = true;
  symbol_table_.add(type_symbol);

  return dict_struct;
}
