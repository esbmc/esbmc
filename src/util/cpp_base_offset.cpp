#include "cpp_base_offset.h"
#include "type_byte_size.h"
#include "cpp_data_object.h"
#include "clang-c-frontend/clang_c_convert.h"

bool cpp_base_offset::offset_to_base(
  const dstring &base_name,
  const typet &type,
  exprt &offset_expr,
  const namespacet &ns)
{
  assert(!has_prefix(base_name, clang_c_convertert::tag_prefix));
  assert(type.is_struct());
  if (base_name == type.tag())
  {
    offset_expr = constant_exprt(0, size_type());
    return false;
  }
  auto result = cpp_data_object::get_data_object(type, base_name, ns);
  BigInt base_offset =
    member_offset(migrate_type(result.first), result.second.name(), &ns);
  offset_expr = constant_exprt(base_offset, size_type());
  return false;
}
