/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/context.h>
#include <util/message/default_message.h>

bool contextt::add(const symbolt &symbol)
{
  std::pair<symbolst::iterator, bool> result =
    symbols.insert(std::pair<irep_idt, symbolt>(symbol.id, symbol));

  if(!result.second)
    return true;

  symbol_base_map.insert(std::pair<irep_idt, irep_idt>(symbol.name, symbol.id));

  ordered_symbols.push_back(&result.first->second);
  return false;
}

bool contextt::move(symbolt &symbol, symbolt *&new_symbol)
{
  symbolt tmp;
  std::pair<symbolst::iterator, bool> result =
    symbols.insert(std::pair<irep_idt, symbolt>(symbol.id, tmp));

  if(!result.second)
  {
    new_symbol = &result.first->second;
    return true;
  }

  symbol_base_map.insert(std::pair<irep_idt, irep_idt>(symbol.name, symbol.id));

  ordered_symbols.push_back(&result.first->second);

  result.first->second.swap(symbol);
  new_symbol = &result.first->second;
  return false;
}

void contextt::dump() const
{
  msg.debug("\nSymbols:");
  // Do assignments based on "value".
  foreach_operand([](const symbolt &s) { s.dump(); });
}

symbolt *contextt::find_symbol(irep_idt name)
{
  auto it = symbols.find(name);
  if(it != symbols.end())
    return &(it->second);
  return nullptr;
}

const symbolt *contextt::find_symbol(irep_idt name) const
{
  auto it = symbols.find(name);
  if(it != symbols.end())
    return &(it->second);
  return nullptr;
}

void contextt::erase_symbol(irep_idt name)
{
  symbolst::iterator it = symbols.find(name);
  if(it == symbols.end())
  {
    msg.error("Couldn't find symbol to erase");
    abort();
  }

  ordered_symbols.erase(
    std::remove_if(
      ordered_symbols.begin(),
      ordered_symbols.end(),
      [&name](const symbolt *s) { return s->id == name; }),
    ordered_symbols.end());
  symbols.erase(it);
}

void contextt::foreach_operand_impl_const(const_symbol_delegate &expr) const
{
  for(const auto &symbol : symbols)
  {
    expr(symbol.second);
  }
}

void contextt::foreach_operand_impl(symbol_delegate &expr)
{
  for(auto &symbol : symbols)
  {
    expr(symbol.second);
  }
}

void contextt::foreach_operand_impl_in_order_const(
  const_symbol_delegate &expr) const
{
  for(auto ordered_symbol : ordered_symbols)
  {
    expr(*ordered_symbol);
  }
}

void contextt::foreach_operand_impl_in_order(symbol_delegate &expr)
{
  for(auto &ordered_symbol : ordered_symbols)
  {
    expr(*ordered_symbol);
  }
}
