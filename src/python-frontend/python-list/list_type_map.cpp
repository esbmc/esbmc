#include "python_list_internal.h"

using namespace python_expr;

std::unordered_map<std::string, std::vector<std::pair<std::string, typet>>>
  python_list::list_type_map{};

void python_list::get_list_type_flags(
  const std::string &list_id,
  const type_handler &th,
  int &type_flag,
  size_t &float_type_id)
{
  type_flag = 0;
  float_type_id = 0;

  bool has_float = false;
  bool has_int = false;
  bool is_string = false;

  size_t map_size = python_list::get_list_type_map_size(list_id);
  for (size_t k = 0; k < map_size; ++k)
  {
    const typet elem_type = python_list::get_list_element_type(list_id, k);
    if (elem_type.is_floatbv())
    {
      if (!has_float)
      {
        float_type_id = std::hash<std::string>{}(th.type_to_string(elem_type));
        has_float = true;
      }
    }
    else if (
      (elem_type.is_pointer() && elem_type.subtype() == char_type()) ||
      (elem_type.is_array() && elem_type.subtype() == char_type()))
    {
      is_string = true;
    }
    else
      has_int = true;
  }

  if (is_string)
    type_flag = 2;
  else if (has_float && has_int)
    type_flag = 3;
  else if (has_float)
    type_flag = 1;
  else
    type_flag = 0;
}

typet python_list::get_list_element_type(
  const std::string &list_id,
  size_t index)
{
  auto type_map_it = list_type_map.find(list_id);

  if (type_map_it == list_type_map.end() || type_map_it->second.empty())
    return typet();

  // If index is out of bounds, return the first element's type
  if (index >= type_map_it->second.size())
    index = 0;

  return type_map_it->second[index].second;
}

std::string
python_list::get_list_element_id(const std::string &list_id, size_t index)
{
  auto it = list_type_map.find(list_id);
  if (it == list_type_map.end() || index >= it->second.size())
    return {};
  return it->second[index].first;
}

typet python_list::check_homogeneous_list_types(
  const std::string &list_id,
  const std::string &func_name)
{
  auto it = list_type_map.find(list_id);

  if (it == list_type_map.end() || it->second.empty())
    return typet();

  const TypeInfo &type_info = it->second;
  size_t list_size = type_info.size();

  // Get the first element's type
  typet elem_type = type_info[0].second;

  // Check whether a type is a string type (char array or char pointer)
  auto is_string_type = [](const typet &t) -> bool {
    return (t.is_array() && t.subtype() == char_type()) ||
           (t.is_pointer() && t.subtype() == char_type());
  };

  // Scan all elements to detect mixed int/float
  bool has_int = elem_type.is_signedbv() || elem_type.is_unsignedbv();
  bool has_float = elem_type.is_floatbv();

  for (size_t i = 1; i < list_size; i++)
  {
    const typet &current_elem_type = type_info[i].second;

    // For string types, all char arrays and char pointers are considered compatible
    if (is_string_type(elem_type) && is_string_type(current_elem_type))
      continue;

    if (current_elem_type.is_floatbv())
      has_float = true;
    else if (
      current_elem_type.is_signedbv() || current_elem_type.is_unsignedbv())
      has_int = true;

    // Only int<->float mixing is allowed (Python promotes int to float).
    // Any other mismatch — including different-width or signed/unsigned integers
    // — is an error.
    bool int_float_mix =
      (elem_type.is_floatbv() && (current_elem_type.is_signedbv() ||
                                  current_elem_type.is_unsignedbv())) ||
      ((elem_type.is_signedbv() || elem_type.is_unsignedbv()) &&
       current_elem_type.is_floatbv());
    if (elem_type != current_elem_type && !int_float_mix)
    {
      throw std::runtime_error(
        "Type mismatch in " + func_name +
        "() call: list contains mixed types. "
        "ESBMC currently requires all elements to have the same type for " +
        func_name + "().");
    }
  }

  // Mixed int and float: Python promotes int to float for comparisons
  if (has_int && has_float)
    return double_type();

  return elem_type;
}

bool python_list::has_mixed_numeric_types(const std::string &list_id)
{
  auto it = list_type_map.find(list_id);
  if (it == list_type_map.end() || it->second.empty())
    return false;
  bool has_int = false, has_float = false;
  for (const auto &elem : it->second)
  {
    if (elem.second.is_floatbv())
      has_float = true;
    else if (elem.second.is_signedbv() || elem.second.is_unsignedbv())
      has_int = true;
  }
  return has_int && has_float;
}

typet python_list::infer_literal_element_type(
  const nlohmann::json &list_literal)
{
  nlohmann::json first_elem = json_utils::get_list_element(list_literal, 0);
  if (first_elem.is_null() || first_elem.empty())
    return typet();

  const type_handler &th = converter_.get_type_handler();

  // A heterogeneous int/float literal is promoted to a homogeneous double list
  // at construction (python_list::get, promote_ints), so every element is a
  // double in __ESBMC_float_buf. Read it as a double regardless of which
  // element the index selects; the first element's int type misreads the bits.
  if (
    list_literal["_type"] == "List" && list_literal.contains("elts") &&
    list_literal["elts"].is_array())
  {
    bool has_int = false, has_float = false;
    for (const auto &e : list_literal["elts"])
    {
      const typet t = th.get_typet(e);
      if (t.is_floatbv())
        has_float = true;
      else if (t.is_signedbv() || t.is_unsignedbv() || t.is_bool())
        has_int = true;
    }
    if (has_int && has_float)
      return double_type();
  }

  return th.get_typet(first_elem);
}

typet python_list::numeric_element_type(const std::string &list_id)
{
  auto it = list_type_map.find(list_id);
  if (it == list_type_map.end() || it->second.empty())
    return typet();

  bool has_float = false;
  const typet first = it->second[0].second;
  for (const auto &elem : it->second)
  {
    const typet &t = elem.second;
    if (t.is_floatbv())
      has_float = true;
    else if (!(t.is_signedbv() || t.is_unsignedbv()))
      return typet(); // non-numeric element: not a numeric list
  }

  // int/float mix (or all-float): Python promotes to float, read as double.
  if (has_float)
    return double_type();

  // All integers: require one shared integer type for a sound single-type read.
  for (const auto &elem : it->second)
    if (elem.second != first)
      return typet();
  return first;
}

void python_list::copy_type_map_entries(
  const std::string &from_list_id,
  const std::string &to_list_id)
{
  auto it = list_type_map.find(from_list_id);
  if (it != list_type_map.end())
  {
    for (const auto &type_entry : it->second)
      list_type_map[to_list_id].push_back(type_entry);
  }
}

size_t python_list::get_list_type_map_size(const std::string &list_id)
{
  auto it = list_type_map.find(list_id);
  if (it == list_type_map.end())
    return 0;
  return it->second.size();
}

void python_list::reverse_type_info(const std::string &list_id)
{
  auto it = list_type_map.find(list_id);
  if (it == list_type_map.end() || it->second.size() <= 1)
    return;
  std::reverse(it->second.begin(), it->second.end());
}
