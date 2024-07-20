#include "cpp_base_offset.h"
#include "type_byte_size.h"

bool cpp_base_offset::offset_to_base(
  const dstring &base_name,
  const typet &type,
  exprt &offset_expr,
  const namespacet &ns)
{
  // get the basename_to_first_component_index map from the type
  assert(type.is_struct());
  struct_typet struct_type = to_struct_type(type);
  struct_type.components();
  irept base_offsets_irept = type.find("base_offsets");
  assert(base_offsets_irept.is_not_nil());
  irept base_offset = base_offsets_irept.find("tag-" + base_name.as_string());
  if (base_offset.is_nil())
  {
    return true;
  }
  else
  {
    uint64_t component_index =
      string2integer(base_offset.id().as_string(), 10).to_uint64();
    // The c++ converter currently creates empty structs with *zero* components.
    // This is actually illegal in C++, all structs must be at least 1 byte in size.
    // We handle this case here until the converter is fixed.
    // TODO: Also handle empty structs that are the last virtual base
    if (
      component_index == struct_type.components().size() &&
      component_index == 0)
    {
      offset_expr = constant_exprt(0, size_type());
      return false;
    }
    BigInt base_offset = member_offset(
      migrate_type(type),
      struct_type.components().at(component_index).name(),
      &ns);
    offset_expr = constant_exprt(base_offset, size_type());
    return false;
  }
}
