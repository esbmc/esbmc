#include <goto-symex/renaming.h>
#include <unordered_map>
#include <utility>
#include <langapi/language_util.h>
#include <irep2/irep2.h>
#include <util/message/message.h>
#include <util/irep/migrate.h>
#include <util/symtab/namespace.h>
#include <util/base/prefix.h>
#include <util/symtab/symbol.h>

namespace
{
/** Treat a symbol with the `__thread` / `_Thread_local` qualifier as a
 *  per-thread instance instead of a shared global. The C frontend already
 *  sets symbolt::is_thread_local from clang::VarDecl::getTLSKind(). */
bool is_thread_local(const namespacet *ns, const irep_idt &name)
{
  if (!ns)
    return false;
  const symbolt *sym = ns->lookup(name);
  return sym && sym->is_thread_local;
}
} // namespace

unsigned renaming::level2t::current_number(const expr2tc &symbol) const
{
  return current_number(name_record(to_symbol2t(symbol)));
}

unsigned renaming::level2t::current_number(const name_record &symbol) const
{
  const valuet *it = current_names.find(symbol);
  return it ? it->count : 0;
}

unsigned int renaming::level1t::current_number(const irep_idt &name) const
{
  const unsigned *it = current_names.find(name_record(name));
  return it ? *it : 0;
}

void renaming::level1t::get_ident_name(expr2tc &sym) const
{
  symbol2t &symbol = to_symbol2t(sym);

  const unsigned *it = current_names.find(name_record(to_symbol2t(sym)));

  if (it == nullptr)
  {
    // Not in this frame's locals. Either a regular global (shared across
    // threads) or a `__thread`-qualified global (per-thread). For TLS,
    // route to level1 with the active thread_id so level2 keys it
    // separately for each thread.
    if (is_thread_local(ns, symbol.thename))
    {
      symbol.rlevel = symbol_renaming_level::level1;
      symbol.level1_num = 0;
      symbol.thread_num = thread_id;
    }
    else
      symbol.rlevel = symbol_renaming_level::level1_global;
    return;
  }

  symbol.rlevel = symbol_renaming_level::level1;
  symbol.level1_num = *it;
  symbol.thread_num = thread_id;
}

void renaming::level2t::get_ident_name(expr2tc &sym) const
{
  symbol2t &symbol = to_symbol2t(sym);

  const valuet *it = current_names.find(name_record(symbol));

  symbol2t::renaming_level lev = symbol.rlevel =
    (symbol.rlevel == symbol_renaming_level::level1)
      ? symbol_renaming_level::level2
      : symbol_renaming_level::level2_global;

  if (it == nullptr)
  {
    // Un-numbered so far.
    symbol.rlevel = lev;
    symbol.level2_num = 0;
    symbol.node_num = 0;
    return;
  }

  symbol.rlevel = lev;
  symbol.level2_num = it->count;
  symbol.node_num = it->node_id;
}

void renaming::level1t::rename(expr2tc &expr)
{
  // rename symbols to their l1 activation-record names (no value substitution)

  if (is_nil_expr(expr))
    return;

  if (is_symbol2t(expr))
  {
    symbol2t &sym = to_symbol2t(expr);

    // first see if it's already an l1 name

    if (sym.rlevel != symbol_renaming_level::level0)
      return;

    const unsigned *it = current_names.find(name_record(sym));

    if (it != nullptr)
    {
      expr = symbol2tc(
        sym.type,
        sym.thename,
        symbol_renaming_level::level1,
        *it,
        0,
        thread_id,
        0);
    }
    else if (is_thread_local(ns, sym.thename))
    {
      // `__thread` global: give each thread its own level1 instance so
      // level2 keys the SSA chain separately per thread.
      expr = symbol2tc(
        sym.type,
        sym.thename,
        symbol_renaming_level::level1,
        0,
        0,
        thread_id,
        0);
    }
    else
    {
      // This isn't an l1 declared name, so it's a global.
      to_symbol2t(expr).rlevel = symbol_renaming_level::level1_global;
    }
  }
  else if (is_address_of2t(expr))
  {
    rename(to_address_of2t(expr).ptr_obj);
  }
  else
  {
    // do this recursively
    expr->Foreach_operand([this](expr2tc &e) { rename(e); });
  }
}

void renaming::level2t::rename(expr2tc &expr)
{
  // rename all the symbols with their last known value
  if (is_nil_expr(expr))
    return;

  if (is_symbol2t(expr))
  {
    symbol2t &sym = to_symbol2t(expr);

    // first see if it's already an l2 name

    if (
      sym.rlevel == symbol_renaming_level::level2 ||
      sym.rlevel == symbol_renaming_level::level2_global)
      return;

    if (sym.thename == "NULL")
      return;
    if (sym.thename == "INVALID")
      return;
    if (has_prefix(sym.thename.as_string(), "nondet$"))
      return;

    const valuet *it = current_names.find(name_record(sym));

    if (it != nullptr)
    {
      // Is this a global symbol? Gets renamed differently.
      symbol2t::renaming_level lev;
      if (
        sym.rlevel == symbol_renaming_level::level0 ||
        sym.rlevel == symbol_renaming_level::level1_global)
        lev = symbol_renaming_level::level2_global;
      else
        lev = symbol_renaming_level::level2;

      if (!is_nil_expr(it->constant))
        expr = it->constant; // sym is now invalid reference
      else
        expr = symbol2tc(
          sym.type,
          sym.thename,
          lev,
          sym.level1_num,
          it->count,
          sym.thread_num,
          it->node_id);
    }
    else
    {
      symbol2t::renaming_level lev;
      if (
        sym.rlevel == symbol_renaming_level::level0 ||
        sym.rlevel == symbol_renaming_level::level1_global)
        lev = symbol_renaming_level::level2_global;
      else
        lev = symbol_renaming_level::level2;

      expr = symbol2tc(
        sym.type, sym.thename, lev, sym.level1_num, 0, sym.thread_num, 0);
    }
  }
  else if (is_address_of2t(expr))
  {
    // do nothing
  }
  else
  {
    // do this recursively
    expr->Foreach_operand([this](expr2tc &e) { rename(e); });
  }
}

void renaming::level2t::coveredinbees(
  expr2tc &lhs_sym,
  unsigned count,
  unsigned node_id)
{
  const symbol_renaming_level lev = to_symbol2t(lhs_sym).rlevel;
  SYMEX_INVARIANT(
    lev == symbol_renaming_level::level1 ||
      lev == symbol_renaming_level::level1_global,
    "L2 assignment counters are keyed by the L1 name");

  const name_record rec(to_symbol2t(lhs_sym));
  current_names.update(rec, [&](valuet entry) {
    // I1: reissuing an index would let two program values share one SSA name.
    SYMEX_INVARIANT(
      entry.count <= count, "L2 assignment counter moved backwards");
    entry.count = count;
    entry.node_id = node_id;
    return entry;
  });
}

namespace
{
/** Rewritten form of every node already visited, keyed by its address. The
 *  original container is kept alongside so a freed node's address cannot be
 *  recycled into a false hit.
 *
 *  A propagated `with` chain over a nested array is a DAG -- each store
 *  references the chain both as the store's source and inside the `index` of
 *  the row it updates -- so an unmemoised walk visits a number of paths
 *  exponential in the store count, and Foreach_operand's detach() gives every
 *  one of them a private copy. */
using original_name_cachet =
  std::unordered_map<const expr2t *, std::pair<expr2tc, expr2tc>>;

void rename_symbol_to_level(expr2tc &expr, symbol2t::renaming_level lev)
{
  if (!is_symbol2t(expr))
    return;

  symbol2t &sym = to_symbol2t(expr);

  // Rename level2_global down to level1_global, not level1
  if (
    lev == symbol_renaming_level::level1 &&
    sym.rlevel == symbol_renaming_level::level2_global)
    lev = symbol_renaming_level::level1_global;
  // level1 and level1_global are equivalent.
  else if (
    lev == symbol_renaming_level::level1 &&
    sym.rlevel == symbol_renaming_level::level1_global)
    return;

  // Can't rename any lower,
  if (sym.rlevel == symbol_renaming_level::level0)
    return;

  // Wipe out some data with default values and set renaming level to whatever
  // was requested.
  switch (lev)
  {
  case symbol_renaming_level::level1:
  case symbol_renaming_level::level1_global:
    sym.rlevel = lev;
    sym.node_num = 0;
    sym.level2_num = 0;
    return;

  case symbol_renaming_level::level0:
    sym.rlevel = lev;
    sym.node_num = 0;
    sym.level2_num = 0;
    sym.thread_num = 0;
    sym.level1_num = 0;
    return;

  default:
    log_error("get_original_nameing to invalid level {}", fmt::underlying(lev));
    abort();
  }
}

void get_original_name_rec(
  expr2tc &expr,
  symbol2t::renaming_level lev,
  original_name_cachet &cache)
{
  if (is_nil_expr(expr))
    return;

  /* Caching an *unshared* node would be a soundness regression, not just a
   * cost: the cache holds the original alive, and that reference makes
   * detach() clone on write, so the node is rebound to a copy instead of
   * rewritten in place. Callers depend on the in-place rewrite -- pinned by
   * unit/goto-symex/overapproximation.test.cpp:319, which goes 0 -> 3 on
   * mentions_invalid_object when the gate is dropped, because an `unknown`
   * surviving into the restored set clears known_exhaustive in
   * dereferencet::dereference and re-opens an invalid_object free variable.
   * An unshared node is reached once anyway, so the gate costs nothing. */
  const expr2t *key = std::as_const(expr).get();
  const bool shared = key->refcount.load(std::memory_order_acquire) > 1;

  if (shared)
  {
    auto cached = cache.find(key);
    if (cached != cache.end())
    {
      expr = cached->second.second;
      return;
    }
  }

  expr2tc original = shared ? expr : expr2tc();

  expr->Foreach_operand(
    [&lev, &cache](expr2tc &e) { get_original_name_rec(e, lev, cache); });
  rename_symbol_to_level(expr, lev);

  if (shared)
    cache.emplace(key, std::make_pair(std::move(original), expr));
}
} // namespace

void renaming::renaming_levelt::get_original_name(
  expr2tc &expr,
  symbol2t::renaming_level lev)
{
  original_name_cachet cache;
  get_original_name_rec(expr, lev, cache);
}

void renaming::level1t::print(std::ostream &out) const
{
  for (const auto &current_name : current_names)
    out << current_name.first.base_name << " --> "
        << "thread " << thread_id << " count " << current_name.second << "\n";
}

void renaming::level2t::print(std::ostream &out) const
{
  for (const auto &current_name : current_names)
  {
    out << current_name.first.base_name;

    if (current_name.first.lev == symbol_renaming_level::level1)
      out << "?" << current_name.first.l1_num << "!"
          << current_name.first.t_num;

    out << " --> ";

    if (!is_nil_expr(current_name.second.constant))
    {
      out << from_expr(
               *migrate_namespace_lookup, "", current_name.second.constant)
          << "\n";
    }
    else
    {
      out << "node " << current_name.second.node_id << " num "
          << current_name.second.count;
      out << "\n";
    }
  }
}

void renaming::level2t::dump() const
{
  std::ostringstream oss;
  print(oss);
  log_debug("rename", "{}", oss.str());
}

void renaming::level2t::make_assignment(
  expr2tc &lhs_symbol,
  const expr2tc &const_value,
  const expr2tc &)
{
  SYMEX_INVARIANT(
    to_symbol2t(lhs_symbol).rlevel == symbol_renaming_level::level1 ||
      to_symbol2t(lhs_symbol).rlevel == symbol_renaming_level::level1_global,
    "L2 assignment counters are keyed by the L1 name");
  const name_record rec(to_symbol2t(lhs_symbol));
  const unsigned expected_count = current_value(rec).count + 1;
  rename(lhs_symbol, expected_count);

  // The rename callee (coveredinbees) re-keyed the same record to
  // expected_count. Fold the confirm-and-store into one HAMT walk (no live
  // reference is held across the mutation): renumber the symbol from the
  // stored generation and record the propagated value.
  symbol2t &symbol = to_symbol2t(lhs_symbol);
  const symbol2t::renaming_level lev =
    (symbol.rlevel == symbol_renaming_level::level0 ||
     symbol.rlevel == symbol_renaming_level::level1_global)
      ? symbol_renaming_level::level2_global
      : symbol_renaming_level::level2;
  current_names.update(rec, [&](valuet entry) {
    SYMEX_INVARIANT(
      entry.count == expected_count,
      "renaming callee bumped a different L2 name record");
    symbol.rlevel = lev;
    symbol.level2_num = entry.count;
    symbol.node_num = entry.node_id;
    entry.constant = const_value;
    return entry;
  });
}

void renaming::level2t::rename_to_record(expr2tc &expr, const name_record &rec)
{
  assert(expr->expr_id == expr2t::symbol_id);
  symbol2t &sym = to_symbol2t(expr);
  assert(sym.thename == rec.base_name);
  assert(sym.rlevel == symbol_renaming_level::level0);

  sym.level1_num = rec.l1_num;
  sym.thread_num = rec.t_num;
  sym.rlevel = rec.lev;
}
