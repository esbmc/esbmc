#ifndef CPROVER_CONTEXT_H
#define CPROVER_CONTEXT_H

#include <map>
#include <util/config.h>
#include <util/symbol.h>
#include <util/type.h>
#include <util/message.h>

typedef std::unordered_map<irep_idt, symbolt, irep_id_hash> symbolst;
typedef std::vector<symbolt *> ordered_symbolst;

typedef std::multimap<irep_idt, irep_idt> symbol_base_mapt;

#define forall_symbol_base_map(it, expr, base_name)                            \
  for(symbol_base_mapt::const_iterator it = (expr).lower_bound(base_name),     \
                                       it_end = (expr).upper_bound(base_name); \
      it != it_end;                                                            \
      it++)

class contextt
{
public:
  symbol_base_mapt symbol_base_map;

  bool add(const symbolt &symbol);
  bool move(symbolt &symbol, symbolt *&new_symbol);

  bool move(symbolt &symbol)
  {
    symbolt *new_symbol;
    return move(symbol, new_symbol);
  }
  symbolt *move_symbol_to_context(symbolt &symbol);

  void clear()
  {
    symbols.clear();
    symbol_base_map.clear();
    ordered_symbols.clear();
  }

  DUMP_METHOD void dump() const;

  void swap(contextt &other)
  {
    symbols.swap(other.symbols);
    symbol_base_map.swap(other.symbol_base_map);
    ordered_symbols.swap(other.ordered_symbols);
  }

  symbolt *find_symbol(irep_idt name);
  const symbolt *find_symbol(irep_idt name) const;

  void erase_symbol(irep_idt name);

  template <typename F>
  void foreach_operand_in_order(F &&f) const
  {
    for(const symbolt *ordered_symbol : ordered_symbols)
      f(*ordered_symbol);
  }

  template <typename F>
  void Foreach_operand_in_order(F &&f)
  {
    for(symbolt *ordered_symbol : ordered_symbols)
      f(*ordered_symbol);
  }

  template <typename F>
  void foreach_operand(F &&f) const
  {
    for(const auto &[id, symbol] : symbols)
      f(symbol);
  }

  template <typename F>
  void Foreach_operand(F &&f)
  {
    for(auto &[id, symbol] : symbols)
      f(symbol);
  }

  size_t size() const
  {
    return symbols.size();
  }

private:
  symbolst symbols;
  ordered_symbolst ordered_symbols;
};

#endif
