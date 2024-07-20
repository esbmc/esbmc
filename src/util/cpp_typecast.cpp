#include "cpp_typecast.h"
#include "clang-cpp-frontend/clang_cpp_convert.h"
#include "c_types.h"
#include "cpp_base_offset.h"
#include "message.h"

void cpp_typecastt::derived_to_base_typecast(
  exprt &expr,
  const typet &dest_type,
  bool is_virtual)
{
  typet src_type = expr.type();
  adjust_pointer_offset(expr, src_type, dest_type, is_virtual);
  expr.make_typecast(dest_type);
}

static void get_vbot_binding_expr_base(exprt &base, exprt &new_expr)
{
  new_expr = dereference_exprt(base, base.type());
  new_expr.set("#lvalue", true);
}

static void get_vbot_binding_expr_vbot_ptr(
  std::string &base_class_id,
  exprt &new_expr,
  const exprt &base_deref)
{
  std::string vbot_type_symb_id =
    clang_cpp_convertert::vbase_offset_type_prefix + base_class_id;
  typet vtable_type = symbol_typet(vbot_type_symb_id);

  std::string vbot_ptr_name =
    base_class_id + "::" + clang_cpp_convertert::vbase_offset_ptr_suffix;
  pointer_typet member_type(vtable_type);
  member_exprt deref_member(base_deref, vbot_ptr_name, member_type);
  deref_member.set("#lvalue", true);

  // we've got the deref type and member. Now we are ready to make the deref new_expr
  new_expr = dereference_exprt(deref_member, member_type);
  new_expr.set("#lvalue", true);
}

void cpp_typecastt::adjust_pointer_offset(
  exprt &expr,
  const typet &src_type,
  const typet &dest_type,
  bool is_virtual)
{
  const typet &src_sub = ns.follow(src_type.subtype());
  const typet &dest_sub = ns.follow(dest_type.subtype());

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
  assert(!src_sub.is_symbol());
  assert(!dest_sub.is_symbol());

  const typet &src_sub_followed = ns.follow(src_sub);

  if (is_virtual)
  {
    if (try_virtual_cast(expr, dest_type, dest_sub_name, src_sub_followed))
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
    if (try_non_virtual_cast(expr, dest_type, dest_sub_name, src_sub_followed))
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

bool cpp_typecastt::try_virtual_cast(
  exprt &expr,
  const typet &dest_type,
  const dstring &dest_sub_name,
  const typet &src_type)
{
  assert(!src_type.is_symbol());
  irept virtual_bases_irept = src_type.find("virtual_bases");
  irept::subt &virtual_bases = virtual_bases_irept.get_sub();
  if (virtual_bases.empty())
  {
    // If src does not have any virtual bases than we can never cast it to dest (at least never virtually).
    // src == dest should be handled already.
    return true;
  }

  size_t count = 0;
  size_t base_index = 0;
  for (size_t index = 0; index < virtual_bases.size(); index++)
  {
    if (virtual_bases[index].type().tag() == dest_sub_name)
    {
      // base_index contains the index of the **first** destination class in the virtual_bases vector.
      // If there are actually multiple destination classes in the virtual_bases vector, we will abort later.
      base_index = index;
      count++;
    }
  }

  if (count == 0 || count > 1)
  {
    return true;
  }

  // Get the destination class as it is in the virtual_bases vector.
  auto dest_base = virtual_bases[base_index];

  // Adjust the pointer by adding the offset
  // (dest_type*) ((char*)source_pointer + offset)
  // where `char* offset = source_pointer->vbo_pointer."base_offset_member"`

  int vbase_offset_index = -static_cast<int>(base_index);
  exprt vbase_offset_index_expr = constant_exprt(
    integer2binary(vbase_offset_index, bv_width(int_type())),
    integer2string(vbase_offset_index),
    int_type());

  std::string source_class_id = expr.type().subtype().identifier().as_string();
  std::string dest_class_id = dest_type.subtype().identifier().as_string();

  // Let's start with base `x` dereferencing
  exprt base_deref;
  get_vbot_binding_expr_base(expr, base_deref);

  // Then deal with X@vtable_pointer dereferencing
  exprt vtable_ptr_deref;
  get_vbot_binding_expr_vbot_ptr(source_class_id, vtable_ptr_deref, base_deref);

  // Then access the `vbase_offsetv` member of the vtable pointer
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

bool cpp_typecastt::try_non_virtual_cast(
  exprt &expr,
  const typet &dest_type,
  const dstring &dest_sub_name,
  const typet &src_type) const
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
