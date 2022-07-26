#include <util/fix_symbol.h>

void fix_symbolt::fix_symbol(symbolt &symbol)
{
  type_mapt::const_iterator it = type_map.find(symbol.id);
  if(it != type_map.end())
    symbol.id = it->second.id();

  replace(symbol.type);
  replace(symbol.value);
}

void fix_symbolt::fix_context(contextt &context)
{
  for(type_mapt::const_iterator t_it = type_map.begin(); t_it != type_map.end();
      t_it++)
  {
    symbolt *symb = context.find_symbol(t_it->first);
    assert(symb != nullptr);

    symbolt s = *symb;
    s.id = t_it->second.identifier();
    context.erase_symbol(t_it->first);
    context.move(s);
  }
}
