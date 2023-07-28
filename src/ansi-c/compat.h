#pragma once

#include <util/irep.h>
#include <util/namespace.h>

// Magic functions that we need because... we want to keep the old frontend
inline void follow_symbol(irept &irep, const namespacet &ns)
{
  while(irep.id() == "symbol")
  {
    const symbolt *symbol = ns.lookup(irep.identifier());
    assert(symbol);

    if(symbol->is_type)
    {
      if(symbol->type.is_nil())
        return;

      irep = symbol->type;
    }
    else
    {
      if(symbol->value.is_nil())
        return;

      irep = symbol->value;
    }
  }
}
