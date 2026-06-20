#include <util/context.h>
#include <util/message.h>
#include <util/message/format.h>

using contextt_detail::by_id;
using contextt_detail::by_order;

bool contextt::add(const symbolt &symbol)
{
  // push_back appends to the sequenced (insertion-order) index; the id index
  // is updated automatically. The bool is false when the id is already present
  // (hashed_unique rejects the duplicate).
  return !symbols.get<by_order>().push_back(symbol).second;
}

bool contextt::move(symbolt &symbol, symbolt *&new_symbol)
{
  auto &by_id_idx = symbols.get<by_id>();

  // If the id is already present, hand back the existing symbol and leave
  // `symbol` untouched — callers (e.g. c_linkt::move -> duplicate()) read the
  // incoming symbol afterwards to merge it. Only move-from `symbol` once we
  // know the insert will succeed.
  auto existing = by_id_idx.find(symbol.id);
  if (existing != by_id_idx.end())
  {
    new_symbol = const_cast<symbolt *>(&*existing);
    return true;
  }

  // Insert the symbol with its final keys directly (the old tmp-then-swap
  // would have mutated the id key after indexing, which multi_index forbids).
  auto result = symbols.get<by_order>().push_back(std::move(symbol));
  new_symbol = const_cast<symbolt *>(&*result.first);
  return false;
}

void contextt::dump() const
{
  log_status("\nSymbols:");
  // Do assignments based on "value".
  foreach_operand([](const symbolt &s) { s.dump(); });
}

symbolt *contextt::find_symbol(irep_idt name)
{
  // NB: the parameter is a symbol id (the table is keyed by symbolt::id).
  auto &by_id_idx = symbols.get<by_id>();
  auto it = by_id_idx.find(name);
  if (it != by_id_idx.end())
    return const_cast<symbolt *>(&*it);
  return nullptr;
}

const symbolt *contextt::find_symbol(irep_idt name) const
{
  const auto &by_id_idx = symbols.get<by_id>();
  auto it = by_id_idx.find(name);
  if (it != by_id_idx.end())
    return &*it;
  return nullptr;
}

void contextt::erase_symbol(irep_idt name)
{
  auto &by_id_idx = symbols.get<by_id>();
  auto it = by_id_idx.find(name);
  if (it == by_id_idx.end())
  {
    log_error("Couldn't find symbol to erase");
    abort();
  }

  // O(1): erasing from one index removes the element from all indices.
  by_id_idx.erase(it);
}

symbolt *contextt::reorder_symbol_to_back(irep_idt name)
{
  auto &by_id_idx = symbols.get<by_id>();
  auto id_it = by_id_idx.find(name);
  if (id_it == by_id_idx.end())
    return nullptr;

  // Move the element to the back of the sequenced (insertion-order) view.
  // relocate is O(1) and does not touch the id index, so the element and any
  // held reference stay valid.
  auto &order = symbols.get<by_order>();
  auto ord_it = symbols.project<by_order>(id_it);
  order.relocate(order.end(), ord_it);
  return const_cast<symbolt *>(&*id_it);
}

void contextt::foreach_operand_impl_const(const_symbol_delegate &expr) const
{
  // Iterate in insertion order. The old unordered_map made foreach_operand
  // hash-ordered, which varied between builds and leaked into GOTO output
  // (global init order, symbol enumeration); the single container lets every
  // walk be deterministic, so foreach_operand now matches the in-order walk.
  for (const symbolt &symbol : symbols.get<by_order>())
    expr(symbol);
}

void contextt::foreach_operand_impl(symbol_delegate &expr)
{
  // Hand out a mutable reference, in insertion order (see the const overload).
  // Callers must not change the id/name keys (they never do — keys are only
  // set on fresh symbols before insertion); mutating type/value/flags in place
  // is safe and leaves the indices intact.
  auto &order = symbols.get<by_order>();
  for (auto it = order.begin(); it != order.end(); ++it)
    expr(const_cast<symbolt &>(*it));
}

void contextt::foreach_operand_impl_in_order_const(
  const_symbol_delegate &expr) const
{
  for (const symbolt &symbol : symbols.get<by_order>())
    expr(symbol);
}

void contextt::foreach_operand_impl_in_order(symbol_delegate &expr)
{
  auto &order = symbols.get<by_order>();
  for (auto it = order.begin(); it != order.end(); ++it)
    expr(const_cast<symbolt &>(*it));
}

symbolt *contextt::move_symbol_to_context(symbolt &symbol)
{
  symbolt *s = find_symbol(symbol.id);
  if (s == nullptr)
  {
    if (move(symbol, s))
    {
      log_error(
        "Couldn't add symbol {} to symbol table\n{}", symbol.name, symbol);
      abort();
    }
    return s;
  }

  // A symbol with this id is already present (s was found by symbol.id).
  // Replace the stored symbol with the incoming one when the latter completes
  // a forward declaration: a function/type definition supplying the missing
  // body/type, or an extern variable becoming a real definition. The incoming
  // symbol shares the id, so the by_id key is unchanged; route the replacement
  // through multi_index's modify() (the in-place reference is const otherwise).
  const bool replace =
    (s->get_type().is_code() && symbol.get_value().is_not_nil() &&
     !s->get_value().is_not_nil()) ||
    (s->is_type && symbol.get_type().is_not_nil() &&
     !s->get_type().is_not_nil()) ||
    (s->is_extern && !symbol.is_extern);

  if (replace)
  {
    auto &by_id_idx = symbols.get<by_id>();
    auto it = by_id_idx.iterator_to(*s);
    by_id_idx.modify(
      it, [&symbol](symbolt &existing) { existing = std::move(symbol); });
    return const_cast<symbolt *>(&*it);
  }
  return s;
}
