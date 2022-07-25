#ifndef ESBMC_UTIL_SYMBOL_GENERATOR_H
#define ESBMC_UTIL_SYMBOL_GENERATOR_H

#include <util/context.h>

class symbol_generator
{
public:
  std::string prefix;
  unsigned counter;

  /**
   * Construct a new symbol generator where the id of every symbol obtained has
   * the given prefix.
   *
   * @param prefix  The id's prefix to give every generated symbol; note: it is
   *                not part of the symbol's name.
   * @param counter Optional value to initialize the counter with; note: the
   *                minimal value appended to a symbol's name and id is
   *                counter+1.
   */
  symbol_generator(std::string prefix, unsigned counter = 0)
    : prefix(std::move(prefix)), counter(counter)
  {
  }

  /**
   * Creates a new symbol with name, id derived from the name and this->prefix,
   * lvalue=true and the given type being set.  The name of the new symbol is
   * derived from the given name_prefix and this->counter in such a way that the
   * obtained id is unique within the given context.
   * This method cannot fail, but it can run into trouble a) under out-of-memory
   * conditions or b) when all possible symbol ids obtainable through the above
   * pattern are already in use (counter is a 'unsigned int').
   *
   * @param context     The context to register this symbol with.
   * @param type        The new symbol's type.
   * @param name_prefix Optional parameter defaulting to the empty string; it is
   *                    not analyzed in any way and used just before appending
   *                    the decimal value of the counter.
   *
   * @return A reference to the new symbol.
   */
  symbolt &new_symbol(
    contextt &context,
    const typet &type,
    const std::string &name_prefix = "");
};

#endif
