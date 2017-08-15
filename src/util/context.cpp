/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/context.h>

bool contextt::add(const symbolt &symbol)
{
  std::pair<symbolst::iterator, bool> result=
    symbols.insert(std::pair<irep_idt, symbolt>(symbol.name, symbol));

  if(!result.second)
    return true;

  symbol_base_map.insert(std::pair<irep_idt, irep_idt>(symbol.base_name, symbol.name));

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

  ordered_symbols.push_back(&result.first->second);

  result.first->second.swap(symbol);
  new_symbol=&result.first->second;
  return false;
}

void contextt::dump() const
{
  std::cout << std::endl << "Symbols:" << std::endl;

  // Do assignments based on "value".
  foreach_operand(
    [] (const symbolt& s)
    {
      s.dump();
    }
  );
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
  if(it == symbols.end())
  {
    std::cerr << "Couldn't find symbol to erase"  << std::endl;
    abort();
  }

  symbols.erase(name);
  ordered_symbols.erase(
    std::remove_if(ordered_symbols.begin(), ordered_symbols.end(),
      [&name](const symbolt *s) { return s->name == name; }),
    ordered_symbols.end());
}

void contextt::foreach_operand_impl_const(const_symbol_delegate& expr) const
{
  for(const auto & symbol : symbols)
  {
    expr(symbol.second);
  }
}

void contextt::foreach_operand_impl(symbol_delegate& expr)
{
  for(auto & symbol : symbols)
  {
    expr(symbol.second);
  }
}

void contextt::foreach_operand_impl_in_order_const(
    const_symbol_delegate& expr) const
{
  for(auto ordered_symbol : ordered_symbols)
  {
    expr(*ordered_symbol);
  }
}

void contextt::foreach_operand_impl_in_order(symbol_delegate& expr)
{
  for(auto & ordered_symbol : ordered_symbols)
  {
    expr(*ordered_symbol);
  }
}

void contextt::remove_unused()
{
  for(auto it = symbols.begin(), ite = symbols.end(); it != ite;)
  {
    if(!it->second.is_used)
      it = symbols.erase(it);
    else
      ++it;
  }

  ordered_symbols.erase(
    std::remove_if(ordered_symbols.begin(), ordered_symbols.end(),
      [](const symbolt *s) { return !s->is_used; }),
    ordered_symbols.end());
}
