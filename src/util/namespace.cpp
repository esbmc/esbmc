#include <cassert>
#include <cstring>
#include <util/namespace.h>
#include <util/message.h>

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

void namespacet::follow_symbol(irept &irep) const
{
  while(irep.id() == "symbol")
  {
    const symbolt *symbol = lookup(irep);
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

typet namespacet::follow(const typet &src, bool deep) const
{
  if(!src.is_symbol())
    return src;

  const symbolt *symbol = lookup(src);

  // let's hope it's not cyclic...
  while(true)
  {
    assert(symbol);
    assert(symbol->is_type);
    if(!symbol->type.is_symbol())
      break;
    symbol = lookup(symbol->type);
  }

  typet res = symbol->type;
  if(!deep)
    return res;

  if(res.id() == "struct" || res.id() == "union")
    for(struct_union_typet::componentt &comp :
        to_struct_union_type(res).components())
      comp.type() = follow(comp.type(), true);

  return res;
}
