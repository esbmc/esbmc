#include "cpp_typecast.h"
#include "clang-cpp-frontend/clang_cpp_convert.h"
#include "util/c_types.h"
#include "util/cpp_base_offset.h"
#include "util/message.h"
#include "util/cpp_data_object.h"

void cpp_typecast::derived_to_base_typecast(
  exprt &expr,
  const typet &dest_type,
  bool is_virtual,
  namespacet &ns)
{
  typet src_type = expr.type();
  adjust_pointer_offset(expr, src_type, dest_type, is_virtual, ns);
  expr.make_typecast(dest_type);
}

void cpp_typecast::get_vbot_binding_expr_base(exprt &base, exprt &new_expr)
{
  assert(base.type().is_pointer());
  assert(base.type().subtype().is_not_nil());
  std::string base_id;
  std::string base_name;
  get_id_name(base.type().subtype(), base_id, base_name);

  typet data_object_symbol_type;
  cpp_data_object::get_data_object_symbol_type(
    base_id, data_object_symbol_type);
  exprt base_deref = dereference_exprt(base, base.type());
  assert(!base_deref.type().is_pointer());
  exprt data_object_base = member_exprt(
    base_deref,
    base_name + cpp_data_object::data_object_suffix,
    data_object_symbol_type);
  data_object_base.set("#lvalue", true);
  new_expr.swap(data_object_base);
}
void cpp_typecast::get_id_name(
  const typet &type,
  std::string &base_id,
  std::string &base_name)
{
  if (type.is_struct())
  {
    base_name = type.tag().as_string();
    base_id = clang_c_convertert::tag_prefix + base_name;
  }
  else if (type.is_symbol())
  {
    base_id = type.identifier().as_string();
    base_name = base_id.substr(clang_c_convertert::tag_prefix.length());
  }
  else
  {
    log_error(
      "cpp-typecast: get_vbot_binding_expr_base: base type is not a struct or "
      "symbol");
    abort();
  }
  assert(!base_id.empty());
  assert(!base_name.empty());
}

static void get_vbot_binding_expr_vbot_ptr(
  std::string &base_class_id,
  exprt &new_expr,
  const exprt &base_deref)
{
  assert(!base_class_id.empty());
  std::string vbot_type_symb_id =
    clang_cpp_convertert::vbase_offset_type_prefix + base_class_id;
  typet vtable_type = symbol_typet(vbot_type_symb_id);

  std::string vbot_ptr_name =
    base_class_id + "::" + clang_cpp_convertert::vbase_offset_ptr_suffix;
  pointer_typet member_type(vtable_type);
  assert(!base_deref.type().is_pointer());
  member_exprt deref_member(base_deref, vbot_ptr_name, member_type);
  deref_member.set("#lvalue", true);

  // we've got the deref type and member. Now we are ready to make the deref new_expr
  new_expr = dereference_exprt(deref_member, member_type);
  new_expr.set("#lvalue", true);
}

void cpp_typecast::adjust_pointer_offset(
  exprt &expr,
  const typet &src_type,
  const typet &dest_type,
  bool is_virtual,
  namespacet &ns)
{
  assert(src_type.is_pointer());
  assert(dest_type.is_pointer());
  const typet &src_sub = ns.follow(src_type.subtype());
  const typet &dest_sub = ns.follow(dest_type.subtype());
  assert(src_sub.is_struct());
  assert(dest_sub.is_struct());

  // Get the names of the source and destination sub types
  dstring src_sub_name = src_sub.get("tag");
  dstring dest_sub_name = dest_sub.get("tag");
  if (src_sub_name.empty() || dest_sub_name.empty())
  {
    // If the tag is empty, then the type is not a struct or union
    log_error(
      "cpp-typecast: type does not appear to be a struct: cast from {} to {}",
      src_sub.pretty(),
      dest_sub.pretty());
    abort();
  }

  if (is_virtual)
  {
    if (try_virtual_cast(expr, dest_type, dest_sub_name, src_sub))
    {
      log_error(
        "cpp-typecast: adjust_pointer_offset: Cannot cast virtually from {} to "
        "{}",
        src_sub_name,
        dest_sub_name);
      abort();
    }
  }
  else
  {
    if (try_non_virtual_cast(expr, dest_type, dest_sub_name, src_sub, ns))
    {
      log_error(
        "cpp-typecast: adjust_pointer_offset: Cannot cast non-virtually from "
        "{} to {}",
        src_sub_name,
        dest_sub_name);
      abort();
    }
  }
}

bool cpp_typecast::try_virtual_cast(
  exprt &expr,
  const typet &dest_type,
  const dstring &dest_sub_name,
  const typet &src_type)
{
  std::string source_class_id;
  std::string dest_class_id;
  std::string ignored;
  get_id_name(expr.type().subtype(), source_class_id, ignored);
  get_id_name(dest_type.subtype(), dest_class_id, ignored);

  // Let's start with base `x` dereferencing
  exprt base_deref;
  get_vbot_binding_expr_base(expr, base_deref);

  // Then deal with X@vtable_pointer dereferencing
  exprt vtable_ptr_deref;
  get_vbot_binding_expr_vbot_ptr(source_class_id, vtable_ptr_deref, base_deref);

  // Then access the `vbase_offsetv` member of the vtable pointer
  assert(!vtable_ptr_deref.type().is_pointer());
  exprt vbase_offset_expr =
    member_exprt(vtable_ptr_deref, dest_sub_name, size_type());

  // Then deal with indexing dereferencing `*(vbase_offsetv + base_offset_index)` aka `vbase_offsetv[base_offset_index]`
  // 1. First cast the pointer to a char pointer
  exprt char_pointer = typecast_exprt(expr, pointer_typet(char_type()));
  // 2. Add the offset
  exprt offset = vbase_offset_expr;
  plus_exprt adjusted_pointer(char_pointer, offset);
  adjusted_pointer.type() = pointer_typet(char_type());
  // 3. Cast the pointer back to the original type
  exprt typecastExprt = typecast_exprt(adjusted_pointer, dest_type);
  expr.swap(typecastExprt);
  return false;
}

bool cpp_typecast::try_non_virtual_cast(
  exprt &expr,
  const typet &dest_type,
  const dstring &dest_sub_name,
  const typet &src_type,
  namespacet &ns)
{
  // Adjust the pointer by adding the offset
  // (dest_type*) ((char*)source_pointer + offset)
  // 1. First cast the pointer to a char pointer
  exprt char_pointer = typecast_exprt(expr, pointer_typet(char_type()));
  // 2. Add the offset
  exprt offset;
  if (cpp_base_offset::offset_to_base(dest_sub_name, src_type, offset, ns))
  {
    return true;
  }
  plus_exprt adjusted_pointer(char_pointer, offset);
  adjusted_pointer.type() = pointer_typet(char_type());
  // 3. Cast the pointer back to the dest src_type
  exprt typecastExprt = typecast_exprt(adjusted_pointer, dest_type);
  expr.swap(typecastExprt);
  return false;
}
