#include <util/lang/python_types.h>
#include <util/irep/std_types.h>

typet none_type()
{
  // Pointer to bool: represents "None"
  return pointer_typet(bool_typet()); // Distinct from any_type
}

typet any_type()
{
  // Pointer to void: represents "Any"
  return pointer_typet(empty_typet()); // void*
}

bool returns_no_value(const typet &t)
{
  return t.is_nil() || t == none_type() || t.id() == "empty";
}

void set_python_aggregate_kind(typet &type, const irep_idt &kind)
{
  type.set(PYTHON_AGGREGATE_ATTR, kind);
}

irep_idt python_aggregate_kind(const typet &type)
{
  return type.get(PYTHON_AGGREGATE_ATTR);
}

bool is_python_internal_aggregate(const typet &type)
{
  return !python_aggregate_kind(type).empty();
}
