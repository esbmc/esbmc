#ifndef CPROVER_CONTEXT_H
#define CPROVER_CONTEXT_H

#include <functional>

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index_container.hpp>

#include <util/config.h>
#include <util/symbol.h>
#include <util/type.h>

/* Symbol table.
 *
 * A single boost::multi_index_container of symbolt holds every symbol and
 * exposes the two views the codebase needs, kept in sync automatically:
 *
 *   - by_id:    hashed_unique on symbolt::id    — O(1) lookup by id
 *   - by_order: sequenced                       — insertion order (init order)
 *
 * multi_index is node-based, so element references/pointers stay valid across
 * insert and erase (only the erased element is invalidated). This replaces the
 * former hand-synced trio (unordered_map + vector<symbolt*> + multimap), which
 * leaked base-name entries on erase, did an O(n) scan to erase from the order
 * vector, and stored raw symbolt* into the map (a documented use-after-free
 * hazard on erase). Same pattern already used by smt_solver_baset::smt_cache.
 *
 * There is no base-name index: the only base-name lookup is the one-shot
 * entry-point search (find `main`) done once per run, served by a linear scan
 * (foreach_named_symbol_with_base) — not worth a third index maintained on
 * every insert/erase.
 */

namespace contextt_detail
{
struct by_id
{
};
struct by_order
{
};

typedef boost::multi_index_container<
  symbolt,
  boost::multi_index::indexed_by<
    boost::multi_index::hashed_unique<
      boost::multi_index::tag<by_id>,
      BOOST_MULTI_INDEX_MEMBER(symbolt, irep_idt, id),
      irep_id_hash>,
    boost::multi_index::sequenced<boost::multi_index::tag<by_order>>>>
  symbol_containert;
} // namespace contextt_detail

class contextt
{
  typedef std::function<void(const symbolt &symbol)> const_symbol_delegate;
  typedef std::function<void(symbolt &symbol)> symbol_delegate;

public:
  explicit contextt()
  {
  }
  ~contextt() = default;
  contextt(const contextt &obj) = delete;
  contextt(contextt &&) noexcept = default;

#ifdef ENABLE_OLD_FRONTEND
  contextt &operator=(const contextt &rhs)
  {
    // copy assignment operator for old frontend typechecking
    if (&rhs == this) // check self assignment
    {
      log_error("Context is copying itself");
    }

    // Since the const messaget& member breaks default copy assignment operation in this class,
    // this no-op copy assignment operator aims to restore the functionality
    // of ansic_c_typecheck function in src/ansi-c/ansi_c_typecheck.cpp prior to commit 4dc8478.
    // However, using a no-op copy assignment operator is quite hacky. We do not recomment it
    // in other places of ESBMC.
    return *this;
  }
#else
  contextt &operator=(const contextt &rhs) = delete;
#endif

  contextt &operator=(contextt &&) noexcept = default;

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
  }

  DUMP_METHOD void dump() const;

  void swap(contextt &other)
  {
    symbols.swap(other.symbols);
  }

  symbolt *find_symbol(const char *name)
  {
    return find_symbol(irep_idt(name));
  }
  symbolt *find_symbol(irep_idt name);
  const symbolt *find_symbol(irep_idt name) const;

  void erase_symbol(irep_idt name);

  /// Move the symbol with id @p name to the end of the insertion-order view,
  /// keeping it in the table. The C/C++ frontends need this when a symbol's
  /// definition must sort after symbols it now depends on (a completed struct
  /// after its field types; a static member in definition rather than
  /// declaration order). O(1); replaces the former erase-and-reinsert, which
  /// copied the symbol and briefly dangled any held symbolt*. Returns the
  /// (stable) symbol, or nullptr if no such id exists.
  symbolt *reorder_symbol_to_back(irep_idt name);

  /// Invoke @p t(id) for every symbol whose base name equals @p base_name, in
  /// insertion order (the former forall_symbol_base_map). @p t returns a bool:
  /// false stops the scan early (the lambda equivalent of `break`). Implemented
  /// as a linear scan: the sole caller is the once-per-run entry-point search,
  /// so a dedicated base-name index is not worth maintaining on every insert.
  template <typename T>
  void foreach_named_symbol_with_base(irep_idt base_name, T &&t) const
  {
    for (const symbolt &s : symbols.get<contextt_detail::by_order>())
      if (s.name == base_name)
        if (!t(s.id))
          break;
  }

  template <typename T>
  void foreach_operand_in_order(T &&t) const
  {
    const_symbol_delegate wrapped(std::cref(t));
    foreach_operand_impl_in_order_const(wrapped);
  }

  template <typename T>
  void Foreach_operand_in_order(T &&t)
  {
    symbol_delegate wrapped(std::ref(t));
    foreach_operand_impl_in_order(wrapped);
  }

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
  contextt_detail::symbol_containert symbols;

  void foreach_operand_impl_const(const_symbol_delegate &expr) const;
  void foreach_operand_impl(symbol_delegate &expr);

  void foreach_operand_impl_in_order_const(const_symbol_delegate &expr) const;
  void foreach_operand_impl_in_order(symbol_delegate &expr);
};

#endif
