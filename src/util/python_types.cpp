#include <util/python_types.h>
#include <util/std_types.h>

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