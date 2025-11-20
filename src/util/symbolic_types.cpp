#include "symbolic_types.h"

/* TODO: this is basically thrash_type_symbols() modulo pointers in slow */
static void complete_type(typet &type, const namespacet &ns)
{
  if (type.is_struct() || type.is_union())
  {
    for (auto &comp : to_struct_union_type(type).components())
      complete_type(comp.type(), ns);
    return;
  }

  if (type.is_array())
    return complete_type(type.subtype(), ns);

  if (type.is_symbol())
  {
    const symbolt *sym = ns.lookup(type.identifier());
    assert(sym);
    assert(sym->is_type);
    assert(!sym->type.is_symbol());
    type = sym->type;
    return complete_type(type, ns);
  }
}

typet get_complete_type(typet type, const namespacet &ns)
{
  complete_type(type, ns);
  return type;
}
