#include "util_irep.h"

#include <util/c_types.h>
#include <util/config.h>

void gen_builtin_type(typet &new_type, Builtin_Type bt)
{
  config.ansi_c.set_data_model(configt::ILP32);
  std::string c_type;
  switch (bt)
  {
  case Builtin_Type::Void:
    new_type = empty_typet();
    c_type = "void";
    break;

  case Builtin_Type::Bool:
    new_type = bool_type();
    c_type = "bool";
    break;

  case Builtin_Type::UChar:
    new_type = unsigned_char_type();
    c_type = "unsigned_char";
    break;

  case Builtin_Type::WChar_U:
    new_type = unsigned_wchar_type();
    c_type = "unsigned_wchar_t";
    break;

  case Builtin_Type::Char16:
    new_type = char16_type();
    c_type = "char16_t";
    break;

  case Builtin_Type::Char32:
    new_type = char32_type();
    c_type = "char32_t";
    break;

  case Builtin_Type::UShort:
    new_type = unsigned_short_int_type();
    c_type = "unsigned_short";
    break;

  case Builtin_Type::UInt:
    new_type = uint_type();
    c_type = "unsigned_int";
    break;

  case Builtin_Type::ULong:
    new_type = long_uint_type();
    c_type = "unsigned_long";
    break;

  case Builtin_Type::ULongLong:
    new_type = long_long_uint_type();
    c_type = "unsigned_long_long";
    break;

  case Builtin_Type::SChar:
    new_type = signed_char_type();
    c_type = "signed_char";
    break;

  case Builtin_Type::WChar_S:
    new_type = wchar_type();
    c_type = "wchar_t";
    break;

  case Builtin_Type::Short:
    new_type = signed_short_int_type();
    c_type = "signed_short";
    break;

  case Builtin_Type::Int:
    new_type = int_type();
    c_type = "signed_int";
    break;

  case Builtin_Type::Long:
    new_type = long_int_type();
    c_type = "signed_long";
    break;

  case Builtin_Type::LongLong:
    new_type = long_long_int_type();
    c_type = "signed_long_long";
    break;

  case Builtin_Type::Half:
    new_type = half_float_type();
    c_type = "_Float16";
    break;

  case Builtin_Type::Float:
    new_type = float_type();
    c_type = "float";
    break;

  case Builtin_Type::Double:
    new_type = double_type();
    c_type = "double";
    break;

  case Builtin_Type::LongDouble:
    new_type = long_double_type();
    c_type = "long_double";
    break;

  default:
    return;
  }

  new_type.set("#cpp_type", c_type);
}

struct_union_typet::componentt gen_component(const char *name, Builtin_Type bt)
{
  typet t;
  gen_builtin_type(t, bt);
  struct_union_typet::componentt comp(name, name, t);
  return comp;
}
