/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_CONTEXT_H
#define CPROVER_CONTEXT_H

#include <iostream>
#include <functional>
#include <map>

#include <hash_cont.h>
#include <type.h>
#include <symbol.h>
#include <string_hash.h>

typedef hash_map_cont<irep_idt, symbolt, irep_id_hash> symbolst;
typedef std::vector<const symbolt*> ordered_symbolst;

typedef std::multimap<irep_idt, irep_idt> symbol_base_mapt;
typedef std::multimap<irep_idt, irep_idt> symbol_module_mapt;

#define forall_symbol_base_map(it, expr, base_name) \
  for(symbol_base_mapt::const_iterator it=(expr).lower_bound(base_name), \
                                       it_end=(expr).upper_bound(base_name); \
      it!=it_end; it++)

#define forall_symbol_module_map(it, expr, module) \
  for(symbol_module_mapt::const_iterator it=(expr).lower_bound(module), \
                                         it_end=(expr).upper_bound(module); \
      it!=it_end; it++)

class contextt
{
  typedef std::function<void (const symbolt &symbol)> const_symbol_delegate;
  typedef std::function<void (symbolt &symbol)> symbol_delegate;

public:
  typedef ::symbolst symbolst;
  typedef ::ordered_symbolst ordered_symbolst;

  symbol_base_mapt symbol_base_map;
  symbol_module_mapt symbol_module_map;

  bool add(const symbolt &symbol);
  bool move(symbolt &symbol, symbolt *&new_symbol);

  bool move(symbolt &symbol)
  { symbolt *new_symbol; return move(symbol, new_symbol); }

  void clear()
  {
    symbols.clear();
    symbol_base_map.clear();
    symbol_module_map.clear();
  }

  void show(std::ostream &out = std::cout) const;

  void swap(contextt &other)
  {
    symbols.swap(other.symbols);
    symbol_base_map.swap(other.symbol_base_map);
    symbol_module_map.swap(other.symbol_module_map);
  }

  symbolt* find_symbol(irep_idt name);
  const symbolt* find_symbol(irep_idt name) const;

  void erase_symbol(irep_idt name);

  template <typename T>
  void foreach_operand(T &&t) const
  {
    const_symbol_delegate wrapped(std::cref(t));
    foreach_operand_impl_const(wrapped);
  }

  template <typename T>
  void Foreach_operand(T &&t)
  {
    symbol_delegate wrapped(std::ref(t));
    foreach_operand_impl(wrapped);
  }

  unsigned int size() const
  {
    return symbols.size();
  }

private:
  symbolst symbols;
  ordered_symbolst ordered_symbols;

  void foreach_operand_impl_const(const_symbol_delegate &expr) const;
  void foreach_operand_impl(symbol_delegate &expr);
};

std::ostream &operator << (std::ostream &out, const contextt &context);

#endif
