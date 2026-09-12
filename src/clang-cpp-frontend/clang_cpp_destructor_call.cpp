#include <clang-cpp-frontend/clang_cpp_destructor_call.h>
#include <util/irep/std_expr.h>
#include <util/message/message.h>

exprt destructor_binding(
  const namespacet &ns,
  const struct_typet &class_type,
  const struct_typet::componentt &dtor,
  const exprt &object)
{
  exprt static_binding("symbol", dtor.type());
  static_binding.identifier(dtor.name());

  if (!dtor.get_bool("is_virtual"))
    return static_binding;

  // The slot is keyed by the destructor's `virtual_name`, i.e. the id of the
  // ultimate overridden destructor. Select the vtable pointer whose table
  // actually carries that slot rather than assuming the class's own vptr comes
  // first among the components.
  const struct_typet::componentt *vptr = nullptr;
  const struct_typet::componentt *slot = nullptr;
  const typet *vtable_type = nullptr;

  for (const auto &comp : class_type.components())
  {
    if (!comp.get_bool("is_vtptr"))
      continue;

    const typet &candidate = ns.follow(comp.type().subtype());
    if (candidate.id() != "struct")
      continue;

    for (const auto &entry : to_struct_type(candidate).components())
      if (entry.get("virtual_name") == dtor.get("virtual_name"))
      {
        vptr = &comp;
        slot = &entry;
        vtable_type = &candidate;
        break;
      }

    if (slot != nullptr)
      break;
  }

  // A class with a virtual destructor always carries a vtable pointer and a
  // matching slot: both are emitted together when the vtable is built. Falling
  // back to the static destructor here would silently skip the derived
  // destructors' side effects, so fail loudly instead.
  if (slot == nullptr)
  {
    log_error(
      "{}: no virtual table slot for destructor `{}` of `{}`",
      __func__,
      dtor.name(),
      class_type.tag());
    abort();
  }

  // *object.@vtable_pointer->~T#
  member_exprt vptr_member(object, vptr->name(), vptr->type());
  dereference_exprt vtable(vptr_member, vptr->type());
  // No further adjust pass runs over this expression, so resolve the vtable
  // symbol type here: member2t requires a resolved struct source.
  vtable.type() = *vtable_type;

  member_exprt slot_member(vtable, slot->name(), slot->type());
  return dereference_exprt(slot_member, slot->type());
}
