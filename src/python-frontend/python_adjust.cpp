#include <python-frontend/python_adjust.h>

#include <irep2/irep2_utils.h>
#include <util/message.h>
#include <vector>

python_adjust::python_adjust(contextt &_context)
  : context(_context), ns(_context)
{
}

bool python_adjust::adjust()
{
  // Hash-table iterators are not stable across mutation, so snapshot the
  // symbol pointers first (mirrors clang_c_adjust::adjust()).
  std::vector<symbolt *> symbol_list;
  context.Foreach_operand_in_order(
    [&symbol_list](symbolt &s) { symbol_list.push_back(&s); });

  bool error = false;
  for (symbolt *symbol : symbol_list)
  {
    if (symbol->is_type)
      continue;

    // Only function bodies carry the member2t/index2t expressions this pass
    // resolves, and only bodies are what goto-convert later migrates via
    // get_value2() (V.4.4b). Reading get_value2() on a data symbol whose value
    // is a by-name-typed constant aggregate would trip constant_struct2t's
    // (un-relaxed) migration assert, so skip non-code symbols.
    if (!is_code_type(symbol->get_type2()))
      continue;

    expr2tc value = symbol->get_value2();
    if (is_nil_expr(value))
      continue;

    const expr2tc original = value;
    adjust_expr(value);
    // Only write back when resolution actually changed the tree. Leaving an
    // unchanged symbol untouched keeps its legacy value cache valid, so
    // goto-convert later sees a byte-identical body (this pass runs *after*
    // clang_cpp_adjust) — the pass is inert until the converter emits transient
    // symbol_type member sources.
    if (value != original)
      symbol->set_value(value);

    // Post-adjust strong invariant (V.1k B.4): re-enforce what the three relaxed
    // construction asserts deferred — no member2t/index2t source and no
    // constant_struct2t type may survive as a transient symbol_type2t. On the
    // live pipeline this pass runs after clang_cpp_adjust, which already resolves
    // every by-name aggregate, so the check never fires and the pass stays inert;
    // it is a dead-but-tested safety net that deterministically catches a
    // B.5-era resolution bug once the pass replaces clang_cpp_adjust and becomes
    // the sole resolver.
    if (has_unresolved_source(value))
    {
      log_error(
        "python_adjust: symbol `{}' retains an unresolved by-name "
        "(symbol_type2t) member/index/struct-literal node after adjust (V.1k "
        "post-adjust invariant violated)",
        symbol->id.as_string());
      error = true;
    }
  }

  return error;
}

namespace
{
// A member2t/index2t source is "resolved" once it is a concrete aggregate the
// strong construction invariant accepts; until then it may carry a transient
// symbol_type2t (the relaxed assert permits this, V.1k step 1).
bool is_resolved_aggregate(const type2tc &t)
{
  return is_struct_type(t) || is_union_type(t) || is_array_type(t) ||
         is_vector_type(t);
}
} // namespace

void python_adjust::adjust_expr(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return;

  // Recurse operands first so nested sources resolve inner-to-outer: building
  // `self.b.a` needs `self.b` already resolved to a struct. Foreach_operand
  // mutates each operand in place, so an inner member2t rebuilt below updates
  // the outer member2t's source before we read its type.
  expr->Foreach_operand([this](expr2tc &op) { adjust_expr(op); });

  // Resolve a transient symbol_type2t member/index source to its followed
  // aggregate, re-establishing the strong source invariant before symex sees
  // the node (the V.1k two-phase invariant: relax at construction, re-enforce
  // here). member2t/index2t are immutable, so rebuild with the resolved source.
  if (is_member2t(expr))
  {
    const member2t &m = to_member2t(expr);
    expr2tc source = m.source_value;
    if (resolve_source(source))
      expr = member2tc(m.type, source, m.member);
  }
  else if (is_index2t(expr))
  {
    const index2t &i = to_index2t(expr);
    expr2tc source = i.source_value;
    if (resolve_source(source))
      expr = index2tc(i.type, source, i.index);
  }
}

bool python_adjust::resolve_source(expr2tc &source)
{
  // A member2t/index2t cannot be constructed over a pointer source (the
  // construction assert rejects pointer_id), so the converter always hands a
  // symbol_type2t-typed source — either a plain symbol2t (the instance) or a
  // dereference2t of a `pointer→tag-Cls` instance pointer, whose result type is
  // the symbol_type pointee. Both reach here as a symbol_type2t source; follow
  // it to the resolved aggregate and retype the node in place (with_type keeps
  // expr2t::type immutable). This is the IREP2-native equivalent of
  // clang_c_adjust's symbol-type completion + pointer auto-deref.
  const type2tc &src_type = source->type;
  if (!is_symbol_type(src_type))
    return false;

  type2tc resolved = ns.follow(src_type);
  if (resolved == src_type || !is_resolved_aggregate(resolved))
    return false;

  source = source->with_type(resolved);
  return true;
}

bool python_adjust::has_unresolved_source(const expr2tc &expr) const
{
  if (is_nil_expr(expr))
    return false;

  // A member/index whose source type is still a symbol_type2t is unresolved:
  // resolve_source could not follow it to a concrete aggregate (e.g. it follows
  // to a non-aggregate scalar), so the strong construction invariant is unmet.
  if (is_member2t(expr) && is_symbol_type(to_member2t(expr).source_value->type))
    return true;
  if (is_index2t(expr) && is_symbol_type(to_index2t(expr).source_value->type))
    return true;
  // A constant_struct2t is the third relaxed construction assert (irep2_expr.h):
  // its own type may be a transient by-name symbol_type2t until the aggregate is
  // followed. Post-adjust it must be a resolved struct too.
  if (is_constant_struct2t(expr) && is_symbol_type(expr->type))
    return true;

  bool found = false;
  expr->foreach_operand([this, &found](const expr2tc &e) {
    if (has_unresolved_source(e))
      found = true;
  });
  return found;
}
