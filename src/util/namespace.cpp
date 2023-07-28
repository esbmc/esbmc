#include <cassert>
#include <cstring>
#include <util/namespace.h>
#include <util/message.h>
#include <irep2/irep2_utils.h>

unsigned namespacet::get_max(const std::string &prefix) const
{
  unsigned max_nr = 0;

  context->foreach_operand([&prefix, &max_nr](const symbolt &s) {
    if(!strncmp(s.id.c_str(), prefix.c_str(), prefix.size()))
      max_nr = std::max(unsigned(atoi(s.id.c_str() + prefix.size())), max_nr);
  });

  return max_nr;
}

const symbolt *namespacet::lookup(const irep_idt &name) const
{
  return context->find_symbol(name);
}


const typet &namespacet::follow(const typet &src) const
{
  if(!src.is_symbol())
    return src;

  const symbolt *symbol = lookup(src.identifier());

  /* If its cyclic it means we don't actually have a definition of this type.
   * In that case we can do nothing and shouldn't even have been called. */

  // let's hope it's not cyclic...
  while(true)
  {
    assert(symbol);
    assert(symbol->is_type);
    if(!symbol->type.is_symbol())
      return symbol->type;
    const symbolt *next = lookup(symbol->type.identifier());
    assert(next != symbol && "cycle of length 1 in ns.follow()");
    symbol = next;
  }
}
