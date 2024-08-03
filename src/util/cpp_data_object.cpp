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
