/*******************************************************************\

Module: Namespace

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <string.h>
#include <assert.h>

#include "namespace.h"

unsigned get_max(const std::string &prefix, const symbolst &symbols)
{
  unsigned max_nr=0;

  forall_symbols(it, symbols)
    if(strncmp(it->first.c_str(), prefix.c_str(), prefix.size())==0)
      max_nr=
        std::max(unsigned(atoi(it->first.c_str()+prefix.size())),
                 max_nr);

  return max_nr;
}

unsigned namespacet::get_max(const std::string &prefix) const
{
  unsigned m=0;

  if(context1!=NULL)
    m=std::max(m, ::get_max(prefix, context1->symbols));

  if(context2!=NULL)
    m=std::max(m, ::get_max(prefix, context2->symbols));

  return m;
}

bool namespacet::lookup(
  const irep_idt &name,
  const symbolt *&symbol) const
{
  symbolst::const_iterator it;

  if(context1!=NULL)
  {
    it=context1->symbols.find(name);

    if(it!=context1->symbols.end())
    {
      symbol=&(it->second);
      return false;
    }
  }

  if(context2!=NULL)
  {
    it=context2->symbols.find(name);

    if(it!=context2->symbols.end())
    {
      symbol=&(it->second);
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
  if(src.id()!="symbol") return src;

  const symbolt *symbol=&lookup(src);

  // let's hope it's not cyclic...
  while(true)
  {
    assert(symbol->is_type);
    if(symbol->type.id()!="symbol") return symbol->type;
    symbol=&lookup(symbol->type);
  }
}

void namespacet::follow_macros(exprt &expr) const
{
  if(expr.id()=="symbol")
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
