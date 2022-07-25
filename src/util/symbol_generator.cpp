#include <util/i2string.h>
#include <util/symbol_generator.h>

symbolt &symbol_generator::new_symbol(
  contextt &context,
  const typet &type,
  const std::string &name_prefix)
{
  symbolt new_symbol;
  symbolt *symbol_ptr;

  do
  {
    new_symbol.name = name_prefix + i2string(++counter);
    new_symbol.id = prefix + id2string(new_symbol.name);
    new_symbol.lvalue = true;
    new_symbol.type = type;
  } while(context.move(new_symbol, symbol_ptr));

  return *symbol_ptr;
}
