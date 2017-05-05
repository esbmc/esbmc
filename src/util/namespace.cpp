/*******************************************************************\

Module: Namespace

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <cstring>
#include <util/namespace.h>

unsigned get_max(const std::string &prefix, const contextt *context)
{
  unsigned max_nr=0;

  context->foreach_operand(
    [&prefix, &max_nr] (const symbolt& s)
    {
      if(!strncmp(s.name.c_str(), prefix.c_str(), prefix.size()))
        max_nr = std::max(unsigned(atoi(s.name.c_str()+prefix.size())), max_nr);
    }
  );

  return max_nr;
}

unsigned namespacet::get_max(const std::string &prefix) const
{
  unsigned m=0;

  if(context1 != nullptr)
    m=std::max(m, ::get_max(prefix, context1));

  if(context2 != nullptr)
    m=std::max(m, ::get_max(prefix, context2));

  return m;
}

bool namespacet::lookup(
  const irep_idt &name,
  const symbolt *&symbol) const
{
  const symbolt* s = nullptr;

  s = context1->find_symbol(name);
  if(s != nullptr)
  {
    symbol = s;
    return false;
  }

  if(context2 != nullptr)
  {
    s = context2->find_symbol(name);
    if(s != nullptr)
    {
      symbol = s;
      return false;
    }
  }

  return true;
}

void namespacet::follow_symbol(irept &irep) const
{
  while(irep.id()=="symbol")
  {
    const symbolt &symbol=lookup(irep);

    if(symbol.is_type)
    {
      if(symbol.type.is_nil())
        return;
      else
        irep=symbol.type;
    }
    else
    {
      if(symbol.value.is_nil())
        return;
      else
        irep=symbol.value;
    }
  }
}

const typet &namespacet::follow(const typet &src) const
{
  if(!src.is_symbol()) return src;

  const symbolt *symbol=&lookup(src);

  // let's hope it's not cyclic...
  while(true)
  {
    assert(symbol->is_type);
    if(!symbol->type.is_symbol()) return symbol->type;
    symbol=&lookup(symbol->type);
  }
}

void namespacet::follow_macros(exprt &expr) const
{
  if(expr.is_symbol())
  {
    const symbolt &symbol=lookup(expr);

    if(symbol.is_macro && !symbol.value.is_nil())
    {
      expr=symbol.value;
      follow_macros(expr);
    }

    return;
  }

  Forall_operands(it, expr)
    follow_macros(*it);
}
