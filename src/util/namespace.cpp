#include <cassert>
#include <cstring>
#include <util/namespace.h>
#include <util/message.h>

unsigned namespacet::get_max(const std::string &prefix) const
{
  unsigned max_nr = 0;

  context->foreach_operand([&prefix, &max_nr](const symbolt &s) {
    if (!strncmp(s.id.c_str(), prefix.c_str(), prefix.size()))
      max_nr = std::max(unsigned(atoi(s.id.c_str() + prefix.size())), max_nr);
  });

  return max_nr;
}

const symbolt *namespacet::lookup(const irep_idt &name) const
{
  return context->find_symbol(name);
}

void namespacet::follow_symbol(irept &irep) const
{
  while (irep.id() == "symbol")
  {
    const symbolt *symbol = lookup(irep);
    assert(symbol);

    if (symbol->is_type)
    {
      if (symbol->type.is_nil())
        return;

      irep = symbol->type;
    }
    else
    {
      if (symbol->value.is_nil())
        return;

      irep = symbol->value;
    }
  }
}

const typet &namespacet::follow(const typet &src) const
{
  if (!src.is_symbol())
    return src;

  const symbolt *symbol = lookup(src);

  /* If its cyclic it means we don't actually have a definition of this type.
   * In that case we can do nothing and shouldn't even have been called. */

  // let's hope it's not cyclic...
  while (true)
  {
    assert(symbol);
    assert(symbol->is_type);
    if (!symbol->type.is_symbol())
      return symbol->type;
    const symbolt *next = lookup(symbol->type);
    assert(next != symbol && "cycle of length 1 in ns.follow()");
    symbol = next;
  }
}
