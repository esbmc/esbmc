#include <python-frontend/type/element_type_registry.h>
#include <python-frontend/type/type_handler.h>

#include <util/lang/c_types.h>

#include <algorithm>
#include <functional>
#include <stdexcept>

namespace
{
bool is_string_type(const typet &t)
{
  return (t.is_array() && t.subtype() == char_type()) ||
         (t.is_pointer() && t.subtype() == char_type());
}

bool is_integer_type(const typet &t)
{
  return t.is_signedbv() || t.is_unsignedbv();
}
} // namespace

void element_type_registry::record(
  const std::string &id,
  const std::string &elem_id,
  const typet &elem_type,
  type_slot slot)
{
  map_for(slot)[id].emplace_back(elem_id, elem_type);
}

void element_type_registry::assign_from(
  const std::string &from,
  const std::string &to,
  type_slot slot)
{
  const entries *source = find(from, slot);
  if (!source)
    return;

  // Copy before touching @p to: inserting it may rehash the map and invalidate
  // @p source, and @p from may be @p to.
  entries copied = *source;
  map_for(slot)[to] = std::move(copied);
}

void element_type_registry::append_from(
  const std::string &from,
  const std::string &to,
  type_slot slot)
{
  const entries *source = find(from, slot);
  if (!source)
    return;

  const entries copied = *source;
  entries &target = map_for(slot)[to];
  target.insert(target.end(), copied.begin(), copied.end());
}

void element_type_registry::pop_last(const std::string &id, type_slot slot)
{
  slot_map &m = map_for(slot);
  auto it = m.find(id);
  if (it == m.end())
    return;

  it->second.pop_back();
  if (it->second.empty())
    m.erase(it);
}

void element_type_registry::reverse(const std::string &id, type_slot slot)
{
  slot_map &m = map_for(slot);
  auto it = m.find(id);
  if (it == m.end() || it->second.size() <= 1)
    return;
  std::reverse(it->second.begin(), it->second.end());
}

const element_type_registry::entries *
element_type_registry::find(const std::string &id, type_slot slot) const
{
  const slot_map &m = map_for(slot);
  auto it = m.find(id);
  if (it == m.end() || it->second.empty())
    return nullptr;
  return &it->second;
}

size_t element_type_registry::size(const std::string &id, type_slot slot) const
{
  const entries *e = find(id, slot);
  return e ? e->size() : 0;
}

typet element_type_registry::element_type(
  const std::string &id,
  size_t index,
  type_slot slot) const
{
  const entries *e = find(id, slot);
  if (!e)
    return typet();

  if (index >= e->size())
    index = 0;

  return (*e)[index].second;
}

typet element_type_registry::last_element_type(
  const std::string &id,
  type_slot slot) const
{
  const entries *e = find(id, slot);
  return e ? e->back().second : typet();
}

std::string element_type_registry::element_id(
  const std::string &id,
  size_t index,
  type_slot slot) const
{
  const entries *e = find(id, slot);
  if (!e || index >= e->size())
    return {};
  return (*e)[index].first;
}

bool element_type_registry::has_mixed_numeric(
  const std::string &id,
  type_slot slot) const
{
  const entries *e = find(id, slot);
  if (!e)
    return false;

  bool has_int = false, has_float = false;
  for (const entry &elem : *e)
  {
    if (elem.second.is_floatbv())
      has_float = true;
    else if (is_integer_type(elem.second))
      has_int = true;
  }
  return has_int && has_float;
}

typet element_type_registry::uniform_element_type(
  const std::string &id,
  type_slot slot) const
{
  const entries *e = find(id, slot);
  if (!e)
    return typet();

  const typet &first = e->front().second;
  const bool uniform =
    std::all_of(e->begin(), e->end(), [&first](const entry &elem) {
      return elem.second == first;
    });
  return uniform ? first : typet();
}

typet element_type_registry::numeric_element_type(
  const std::string &id,
  type_slot slot) const
{
  const entries *e = find(id, slot);
  if (!e)
    return typet();

  bool has_float = false;
  for (const entry &elem : *e)
  {
    if (elem.second.is_floatbv())
      has_float = true;
    else if (!is_integer_type(elem.second))
      return typet(); // non-numeric element: not a numeric list
  }

  // int/float mix (or all-float): Python promotes to float, read as double.
  if (has_float)
    return double_type();

  // All integers: require one shared integer type for a sound single-type read.
  return uniform_element_type(id, slot);
}

typet element_type_registry::homogeneous_element_type(
  const std::string &id,
  const std::string &func_name,
  type_slot slot) const
{
  const entries *e = find(id, slot);
  if (!e)
    return typet();

  const typet elem_type = e->front().second;

  bool has_int = is_integer_type(elem_type);
  bool has_float = elem_type.is_floatbv();

  for (size_t i = 1; i < e->size(); i++)
  {
    const typet &current_elem_type = (*e)[i].second;

    // All char arrays and char pointers are compatible string types.
    if (is_string_type(elem_type) && is_string_type(current_elem_type))
      continue;

    if (current_elem_type.is_floatbv())
      has_float = true;
    else if (is_integer_type(current_elem_type))
      has_int = true;

    // Only int<->float mixing is allowed (Python promotes int to float).
    // Any other mismatch — including different-width or signed/unsigned
    // integers — is an error.
    const bool int_float_mix =
      (elem_type.is_floatbv() && is_integer_type(current_elem_type)) ||
      (is_integer_type(elem_type) && current_elem_type.is_floatbv());
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

typet element_type_registry::memoized_pop_type(const std::string &site) const
{
  auto it = pop_memo_.find(site);
  return it != pop_memo_.end() ? it->second : typet();
}

void element_type_registry::memoize_pop_type(
  const std::string &site,
  const typet &elem_type)
{
  pop_memo_[site] = elem_type;
}

void element_type_registry::type_flags(
  const std::string &id,
  const type_handler &th,
  int &type_flag,
  size_t &float_type_id,
  type_slot slot) const
{
  type_flag = 0;
  float_type_id = 0;

  bool has_float = false;
  bool has_int = false;
  bool is_string = false;

  const entries *e = find(id, slot);
  for (size_t k = 0; e && k < e->size(); ++k)
  {
    const typet &elem_type = (*e)[k].second;
    if (elem_type.is_floatbv())
    {
      if (!has_float)
      {
        float_type_id = std::hash<std::string>{}(th.type_to_string(elem_type));
        has_float = true;
      }
    }
    else if (is_string_type(elem_type))
      is_string = true;
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
