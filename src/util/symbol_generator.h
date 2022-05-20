
#ifndef ESBMC_UTIL_SYMBOL_GENERATOR_H
#define ESBMC_UTIL_SYMBOL_GENERATOR_H

#include <util/context.h>

class symbol_generator
{
public:
  std::string prefix;
  unsigned counter;

  symbol_generator(std::string prefix, unsigned counter = 0)
    : prefix(std::move(prefix)), counter(counter)
  {
  }

  symbolt &new_symbol(
    contextt &context,
    const typet &type,
    const std::string &name_prefix = "");
};

#endif
