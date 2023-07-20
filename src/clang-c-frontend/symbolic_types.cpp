#include "symbolic_types.h"

/* TODO: this is basically thrash_type_symbols() modulo pointers in slow */
bool get_complete_type(typet &type, const namespacet &ns)
{
  if(type.is_struct() || type.is_union())
  {
    bool r = false;
    for(auto &comp : to_struct_union_type(type).components())
      r |= get_complete_type(comp.type(), ns);
    return r;
  }
  else if(type.is_array())
  {
    return get_complete_type(type.subtype(), ns);
  }
  else if(type.is_symbol())
  {
    const symbolt *complete_type = ns.lookup(type.identifier());
    assert(complete_type);
    assert(complete_type->is_type);
    assert(!complete_type->type.is_symbol());
    type = complete_type->type;
    get_complete_type(type, ns);
    return true;
  }
  return false;
}

bool contains_symbolic_struct_types(
  const typet &type,
  typet &complete_type,
  const namespacet &ns)
{
  complete_type = type;
  return get_complete_type(complete_type, ns);
}
