/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "context.h"

bool contextt::add(const symbolt &symbol)
{
  std::pair<symbolst::iterator, bool> result=
    symbols.insert(std::pair<irep_idt, symbolt>(symbol.name, symbol));

  if(!result.second)
    return true;

  symbol_base_map.insert(std::pair<irep_idt, irep_idt>(symbol.base_name, symbol.name));
  symbol_module_map.insert(std::pair<irep_idt, irep_idt>(symbol.module, symbol.name));

  ordered_symbols.push_back(&result.first->second);
  return false;
}

bool contextt::move(symbolt &symbol, symbolt *&new_symbol)
{
  symbolt tmp;

  std::pair<symbolst::iterator, bool> result=
    symbols.insert(std::pair<irep_idt, symbolt>(symbol.name, tmp));

  if(!result.second)
  {
    new_symbol = &result.first->second;
    return true;
  }

  symbol_base_map.insert(std::pair<irep_idt, irep_idt>(symbol.base_name, symbol.name));
  symbol_module_map.insert(std::pair<irep_idt, irep_idt>(symbol.module, symbol.name));

  ordered_symbols.push_back(&result.first->second);

  result.first->second.swap(symbol);
  new_symbol=&result.first->second;
  return false;
}

void contextt::show(std::ostream &out) const
{
  out << std::endl << "Symbols:" << std::endl;

  // Do assignments based on "value".
  foreach_operand(
    [] (const symbolt& s)
    {
      s.dump();
    }
  );
}

std::ostream &operator << (std::ostream &out, const contextt &context)
{
  context.show(out);
  return out;
}

symbolt* contextt::find_symbol(irep_idt name)
{
  symbolst::iterator it = symbols.find(name);

  if(it != symbols.end())
    return &(it->second);

  return nullptr;
}

const symbolt* contextt::find_symbol(irep_idt name) const
{
  symbolst::const_iterator it = symbols.find(name);

  if(it != symbols.end())
    return &(it->second);

  return nullptr;
}

void contextt::erase_symbol(irep_idt name)
{
  symbolst::iterator it = symbols.find(name);
  assert(it != symbols.end());

  symbols.erase(name);
  ordered_symbols.erase(
    std::remove_if(ordered_symbols.begin(), ordered_symbols.end(),
      [&name](const symbolt *s) { return s->name == name; }),
    ordered_symbols.end());
}

void contextt::foreach_operand_impl_const(const_symbol_delegate& expr) const
{
  for(symbolst::const_iterator it = symbols.begin();
      it != symbols.end();
      it++)
  {
    expr(it->second);
  }
}

void contextt::foreach_operand_impl(symbol_delegate& expr)
{
  for(symbolst::iterator it = symbols.begin();
      it != symbols.end();
      it++)
  {
    expr(it->second);
  }
}
