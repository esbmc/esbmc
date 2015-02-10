/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_CONTEXT_H
#define CPROVER_CONTEXT_H

#include <iostream>

#include <map>

#include <hash_cont.h>
#include <type.h>
#include <symbol.h>
#include <string_hash.h>

typedef hash_map_cont<irep_idt, symbolt, irep_id_hash> symbolst;

#define forall_symbols(it, expr) \
  for(symbolst::const_iterator it=(expr).begin(); \
      it!=(expr).end(); it++)

#define Forall_symbols(it, expr) \
  for(symbolst::iterator it=(expr).begin(); \
      it!=(expr).end(); it++)

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
public:
  typedef ::symbolst symbolst;

  symbolst symbols;
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

  const irept &value(const irep_idt &name) const;

  void swap(contextt &other)
  {
    symbols.swap(other.symbols);
    symbol_base_map.swap(other.symbol_base_map);
    symbol_module_map.swap(other.symbol_module_map);
  }

  bool has_symbol(const irep_idt &name) const
  {
    return symbols.find(name)!=symbols.end();
  }
};

std::ostream &operator << (std::ostream &out, const contextt &context);

#endif
