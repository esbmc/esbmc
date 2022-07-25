#include <algorithm>
#include <util/i2string.h>
#include <util/rename.h>

void get_new_name(symbolt &symbol, const namespacet &ns)
{
  get_new_name(symbol.id, ns);
}

void get_new_name(irep_idt &new_name, const namespacet &ns)
{
  if(!ns.lookup(new_name))
    return;

  std::string prefix = id2string(new_name) + "_";

  new_name = prefix + i2string(ns.get_max(prefix) + 1);
}
