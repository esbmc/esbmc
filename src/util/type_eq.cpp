#include <cassert>
#include <util/type_eq.h>

bool type_eq(const typet &type1, const typet &type2, const namespacet &ns)
{
  if(type1 == type2)
    return true;

  if(type1.id() == "symbol")
  {
    const symbolt *symbol = ns.lookup(type1);
    assert(symbol);
    if(!symbol->is_type)
      throw "symbol " + id2string(symbol->name) + " is not a type";

    return type_eq(symbol->type, type2, ns);
  }

  if(type2.id() == "symbol")
  {
    const symbolt *symbol = ns.lookup(type2);
    assert(symbol);
    if(!symbol->is_type)
      throw "symbol " + id2string(symbol->name) + " is not a type";

    return type_eq(type1, symbol->type, ns);
  }

  return false;
}
