/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "context.h"

const irept &contextt::value(const irep_idt &name) const
{
  symbolst::const_iterator it=symbols.find(name);
  if(it==symbols.end()) return get_nil_irep();
  return it->second.value;
}

bool contextt::add(const symbolt &symbol)
{
  if(!symbols.insert(std::pair<irep_idt, symbolt>(symbol.name, symbol)).second)
    return true;
    
  symbol_base_map.insert(std::pair<irep_idt, irep_idt>(symbol.base_name, symbol.name));
  symbol_module_map.insert(std::pair<irep_idt, irep_idt>(symbol.module, symbol.name));

  return false;
}

bool contextt::move(symbolt &symbol, symbolt *&new_symbol)
{
  symbolt tmp;

  std::pair<symbolst::iterator, bool> result=
    symbols.insert(std::pair<irep_idt, symbolt>(symbol.name, tmp));

  if(!result.second)
  {
    new_symbol=&result.first->second;
    return true;
  }
    
  symbol_base_map.insert(std::pair<irep_idt, irep_idt>(symbol.base_name, symbol.name));
  symbol_module_map.insert(std::pair<irep_idt, irep_idt>(symbol.module, symbol.name));

  result.first->second.swap(symbol);
  new_symbol=&result.first->second;

  return false;
}

void contextt::show(std::ostream &out) const
{
  out << std::endl << "Symbols:" << std::endl;

  forall_symbols(it, symbols)
    out << it->second;
}

std::ostream &operator << (std::ostream &out, const contextt &context)
{
  context.show(out);
  return out;
}
