#include <python-frontend/python_adjust.h>
#include <irep2/irep2_utils.h>
#include <util/message.h>
#include <util/std_expr.h>

python_adjust::python_adjust(contextt &_context)
  : context(_context), ns(_context)
{
}

namespace
{
// Cheap legacy-exprt pre-scan, run before migrating a symbol value to IREP2.
// The adjuster only acts on member/index nodes whose aggregate source is an
// unresolved by-name `symbol` type (the V.1k transient pre-resolution state it
// follows to a struct/union/array). A symbol whose legacy value carries no such
// node has nothing to resolve, so migrating it is pure overhead.
bool legacy_needs_adjust(const exprt &e)
{
  if (
    (e.is_member() || e.is_index()) && !e.operands().empty() &&
    e.op0().type().id() == typet::t_symbol)
    return true;

  forall_operands (it, e)
    if (legacy_needs_adjust(*it))
      return true;

  return false;
}

// True if migrating @p e to IREP2 would violate a construction invariant. The
// converter-emitted transient sources this pass targets are simple (a member2t/
// index2t over a symbol_type2t symbol) and always migrate-safe. Model-library
// bodies (e.g. the `all`/`from_bytes` builtins) are not — they embed shapes the
// IREP2 constructors reject before this pass can resolve anything:
//   - a constant aggregate literal still carrying a by-name `symbol` type
//     (constant_struct2t/constant_union2t require a concrete struct/union);
//   - `ptr[i]` indexing, where index2t rejects a pointer source (the legacy node
//     is lowered to a dereference after the frontend, post-migrate).
// (kept in sync with converter_binop.cpp's migrate_unsafe_operand.)
// Such bodies do not need this pass — the legacy path resolves them downstream —
// so skipping them is sound and keeps the eager migration off the rocks.
bool migrate_unsafe(const exprt &e)
{
  if (
    (e.id() == typet::t_struct || e.id() == typet::t_union) &&
    e.type().id() == typet::t_symbol)
    return true;

  if (e.is_index() && !e.operands().empty() && e.op0().type().is_pointer())
    return true;

  forall_operands (it, e)
    if (migrate_unsafe(*it))
      return true;

  return false;
}
} // namespace

bool python_adjust::adjust()
{
  // warning! hash-table iterators are not stable — snapshot first, exactly as
  // clang_c_adjust::adjust() does.
  symbol_listt symbol_list;
  context.Foreach_operand_in_order(
    [&symbol_list](symbolt &s) { symbol_list.push_back(&s); });

  bool error = false;
  Forall_symbol_list(it, symbol_list)
  {
    symbolt &symbol = **it;
    if (symbol.is_type)
      continue;

    // Python-only by design (V.1k RV-adj4): only the Python converter emits the
    // pre-adjust symbol_type2t member/index sources this pass resolves, so leave
    // the C operational-model bodies to the legacy path.
    if (symbol.mode != "Python")
      continue;

    // Skip symbols with nothing to resolve, and symbols whose value cannot be
    // safely migrated (model-library bodies carrying tag-typed aggregates or
    // pointer-indexing — see migrate_unsafe). The genuine targets this pass
    // resolves are migrate-safe by construction; the rest are handled by the
    // legacy path downstream.
    const exprt &legacy_value = symbol.get_value();
    if (!legacy_needs_adjust(legacy_value) || migrate_unsafe(legacy_value))
      continue;

    expr2tc value = symbol.get_value2();
    if (is_nil_expr(value))
      continue;

    adjust_expr(value);
    symbol.set_value(value);

    // Post-adjust strong invariant (V.1k B.4): re-enforce that no member/index
    // source survived as an unresolved symbol_type2t. The construction asserts
    // permit that source only as the transient pre-resolution state this pass
    // discharges; a survivor (e.g. an unregistered tag) is an internal error.
    if (has_unresolved_source(value))
    {
      log_error(
        "python_adjust: symbol `{}' retains an unresolved member/index source "
        "after adjust (V.1k post-adjust invariant violated)",
        symbol.id.as_string());
      error = true;
    }
  }

  return error;
}

void python_adjust::adjust_expr(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return;

  // Resolve sub-expressions first: a source must be resolved before the
  // member/index that reads it (resolve `X` before building `X.a`).
  expr->Foreach_operand([this](expr2tc &e) { adjust_expr(e); });

  if (is_member2t(expr))
  {
    const member2t &m = to_member2t(expr);
    expr2tc source = m.source_value;
    if (resolve_aggregate_source(source))
      expr = member2tc(expr->type, source, m.member);
  }
  else if (is_index2t(expr))
  {
    const index2t &i = to_index2t(expr);
    expr2tc source = i.source_value;
    if (resolve_aggregate_source(source))
      expr = index2tc(expr->type, source, i.index);
  }
}

bool python_adjust::resolve_aggregate_source(expr2tc &source) const
{
  // A member2t/index2t source reaches the adjuster as an unresolved by-name
  // symbol_type2t (the relaxed-assert transient state); follow it to its
  // struct/union/array. A pointer base never appears here: the member2t/index2t
  // construction assert forbids a pointer source, so the converter dereferences
  // it (yielding a symbol_type2t-typed source) before building the node.
  if (!is_symbol_type(source->type))
    return false;

  // ns.follow() asserts (debug) / null-derefs (release) on an unregistered tag.
  // Leave such a source untouched: the post-adjust strong-invariant check (B.4)
  // is the correct place to flag a tag that never resolved.
  if (context.find_symbol(to_symbol_type(source->type).symbol_name) == nullptr)
    return false;

  source = source->with_type(ns.follow(source->type));
  return true;
}

bool python_adjust::has_unresolved_source(const expr2tc &expr) const
{
  if (is_nil_expr(expr))
    return false;

  if (is_member2t(expr) && is_symbol_type(to_member2t(expr).source_value->type))
    return true;
  if (is_index2t(expr) && is_symbol_type(to_index2t(expr).source_value->type))
    return true;

  bool found = false;
  expr->foreach_operand([this, &found](const expr2tc &e) {
    if (has_unresolved_source(e))
      found = true;
  });
  return found;
}
