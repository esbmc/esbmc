#include <python-frontend/python_adjust.h>
#include <irep2/irep2_utils.h>
#include <util/message.h>

python_adjust::python_adjust(contextt &_context)
  : context(_context), ns(_context)
{
}

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

    // Only adjust symbols whose IREP2 value is authoritative: those are the
    // converter's IREP2-native member/index sources this pass exists to
    // resolve. Forcing get_value2() on a legacy-valued symbol (e.g. an
    // operational-model function body) would migrate sub-expressions the
    // frontend left with unresolved struct tags -- a latent hole the lazy
    // legacy/IREP2 split tolerates precisely because nothing reads their IREP2
    // side (see symbolt). Until B.3 makes the converter emit these natively the
    // pass is therefore a true no-op on today's corpus, as documented.
    if (!symbol.has_native_value2())
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
