#include "cpp_base_offset.h"
#include "type_byte_size.h"
#include "cpp_data_object.h"

bool cpp_base_offset::offset_to_base(
  const dstring &base_name,
  const typet &type,
  exprt &offset_expr,
  const namespacet &ns)
{
  assert(type.is_struct());
  if (base_name == type.tag())
  {
    offset_expr = constant_exprt(0, size_type());
    return false;
  }
  struct_typet struct_type = to_struct_type(type);
  bool found = false;
  for (const auto &component : struct_type.components())
  {
    if (!has_suffix(component.name(), cpp_data_object::data_object_suffix))
    {
      continue;
    }
    const dstring &data_object_name = component.name();
    const typet &data_object_type = component.type();
    if (
      data_object_name ==
      type.tag().as_string() + cpp_data_object::data_object_suffix)
    {
      const struct_typet &data_object_type_followed =
        to_struct_type(ns.follow(data_object_type));
      for (const auto &comp : data_object_type_followed.components())
      {
        if (
          comp.name() ==
          base_name.as_string() + cpp_data_object::data_object_suffix)
        {
          BigInt base_offset = member_offset(
            migrate_type(data_object_type_followed), comp.name(), &ns);
          offset_expr = constant_exprt(base_offset, size_type());
          found = true;
          break;
        }
      }
    }
    else if (
      data_object_name ==
      base_name.as_string() + cpp_data_object::data_object_suffix)
    {
      assert(!found);
      BigInt base_offset =
        member_offset(migrate_type(ns.follow(type)), data_object_name, &ns);
      offset_expr = constant_exprt(base_offset, size_type());
      found = true;
      break;
    }
  }
  assert(found);
  return false;
}
