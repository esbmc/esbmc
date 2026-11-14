#include <util/symtab/fix_symbol.h>

void fix_symbolt::fix_symbol(symbolt &symbol)
{
  type_mapt::const_iterator it = type_map.find(symbol.id);
  if (it != type_map.end())
    symbol.id = it->second.id();

  // replace() writes through every node it visits, which unshares shared
  // types across the whole symbol even when there is nothing to replace.
  if (type_map.empty() && expr_map.empty())
    return;

  typet t = symbol.get_type();
  replace(t);
  symbol.set_type(std::move(t));
  exprt v = symbol.get_value();
  replace(v);
  symbol.set_value(std::move(v));
}

void fix_symbolt::fix_context(contextt &context)
{
  for (type_mapt::const_iterator t_it = type_map.begin();
       t_it != type_map.end();
       ++t_it)
  {
    symbolt *symb = context.find_symbol(t_it->first);
    assert(symb != nullptr);

    symbolt s = *symb;
    s.id = t_it->second.identifier();
    context.erase_symbol(t_it->first);
    context.move(s);
  }
}
