#include "cpp_data_object.h"
#include <clang-c-frontend/clang_c_convert.h>

struct_typet &cpp_data_object::get_data_object_type(
  const std::string &class_name,
  contextt &context)
{
  assert(class_name.find(clang_c_convertert::tag_prefix) == 0);
  typet symbol_type;
  get_data_object_symbol_type(class_name, symbol_type);
  assert(symbol_type.is_symbol());
  symbolt *s = context.find_symbol(symbol_type.identifier());
  assert(s);
  assert(s->type.is_struct());
  return to_struct_type(s->type);
}
void cpp_data_object::get_data_object_symbol_type(
  const std::string &class_name,
  typet &data_object_symbol_type)
{
  assert(class_name.find(clang_c_convertert::tag_prefix) == 0);

  typet symbol_type =
    symbol_typet(class_name + cpp_data_object::data_object_suffix);
  assert(symbol_type.is_symbol());
  data_object_symbol_type = symbol_type;
}
std::pair<const typet &, struct_union_typet::componentt>
cpp_data_object::get_data_object(
  const typet &type,
  const dstring &data_object_name,
  const namespacet &ns)
{
  struct_typet struct_type = to_struct_type(type);
  bool found = false;
  for (const auto &component : struct_type.components())
  {
    if (!has_suffix(component.name(), cpp_data_object::data_object_suffix))
    {
      continue;
    }
    const dstring &own_data_object_name = component.name();
    const typet &data_object_type = component.type();
    if (
      own_data_object_name ==
      type.tag().as_string() + cpp_data_object::data_object_suffix)
    {
      // This is the data object for `type` itself that contains all non-virtual base data objects
      // Recurse to find the component in the data object
      const struct_typet &data_object_type_followed =
        to_struct_type(ns.follow(data_object_type));
      for (const auto &comp : data_object_type_followed.components())
      {
        if (
          (comp.name() == data_object_name.as_string() +
                            cpp_data_object::data_object_suffix) ||
          (comp.name() == data_object_name && comp.get_bool("#is_c_like")))
        {
          return {data_object_type_followed, comp};
        }
      }
    }
    else if (
      own_data_object_name ==
      data_object_name.as_string() + cpp_data_object::data_object_suffix)
    {
      // This is the data object for a virtual base
      return {type, component};
    }
  }
  abort();
}
