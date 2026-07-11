#include <python-frontend/python_adjust.h>

#include <clang-c-frontend/padding.h>
#include <irep2/irep2_utils.h>
#include <util/message.h>
#include <util/migrate.h>
#include <util/prefix.h>
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
    // constant_struct2t type may survive as a transient symbol_type2t, and a
    // resolved-struct literal must carry one operand per component. Pre-S2
    // this fired on every flag-on run (the OM exception literals,
    // docs/irep2-migration.md "S1 outcome" finding 2); S2's aggregate-literal
    // completion drained those, so a firing now means a node shape the
    // remaining S-steps (S3+) must resolve — the per-node detail below is
    // that work-list.
    std::vector<std::string> unresolved;
    collect_unresolved_sources(value, unresolved);
    if (!unresolved.empty())
    {
      log_error(
        "python_adjust: symbol `{}' retains {} unresolved by-name "
        "(symbol_type2t) node(s) after adjust (V.1k post-adjust invariant "
        "violated):",
        symbol->id.as_string(),
        unresolved.size());
      for (const std::string &entry : unresolved)
        log_error("  {}", entry);
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

// The four reserved pad-member names add_padding assigns (padding.cpp). The
// exact-prefix match matters: `$` cannot appear in a Python identifier, but
// OM struct tags originate from C/C++ headers where Clang accepts `$` as an
// extension — a substring test could misfire on a legitimate member.
bool is_padding_member_name(const std::string &name)
{
  return has_prefix(name, "anon_pad$") ||
         has_prefix(name, "anon_bit_field_pad$") ||
         has_prefix(name, "ext_int_pad$") || name == "$pad";
}
} // namespace

void python_adjust::adjust_expr(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return;

  // Complete the node's own type first (macro expansion, array size, struct
  // padding), mirroring the legacy adjust_expr's leading
  // `adjust_type(expr.type())`. expr2t::type is immutable, so rebuild via
  // with_type only when completion changed it — on the live (post-
  // clang_cpp_adjust) pipeline every body type is already complete, so this
  // is a no-op and the pass stays inert. Caveat: with_type aborts on kinds
  // with no substitutable type slot (constant_bool, relations, bool ops —
  // irep2_expr.cpp); those all carry scalar types adjust_type never changes,
  // so the rebuild is unreachable for them. An adjust_type arm that starts
  // rewriting scalar types must revisit this. A by-name constant_struct2t is
  // excluded: a bare with_type retype would skip the S2 arm's padding-operand
  // insertion (a macro tag would expand here), leaving a literal whose
  // operand count silently disagrees with its resolved type — the S2 arm
  // below owns that node shape entirely.
  if (!(is_constant_struct2t(expr) && is_symbol_type(expr->type)))
  {
    type2tc t = expr->type;
    adjust_type(t);
    if (t != expr->type)
      expr = expr->with_type(t);
  }

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
  else if (is_constant_struct2t(expr) && is_symbol_type(expr->type))
  {
    // S2: aggregate-literal completion — the third relaxed construction
    // assert. The legacy adjust_struct (clang_c_adjust_expr.cpp:152-176)
    // follows the type only to read components, inserts padding operands and
    // leaves the literal's own type lazily by-name; IREP2's strong invariant
    // requires the resolved type on the node, so this arm resolves eagerly
    // (the RV-adj6 divergence, understood and deliberate). On today's
    // pipeline the by-name survivors are the OM exception literals
    // (raise IndexError(...) et al., docs/irep2-migration.md "S1 outcome"
    // finding 2): their operands were already padded by the legacy pass, so
    // only the retype fires; the padding-operand insertion below completes a
    // converter-built literal once the flip makes this pass the sole
    // resolver.
    // Guard the follow: ns.follow asserts on an unknown tag, but an
    // unresolvable literal must instead survive to the exit invariant
    // (mirrors the top-level-symbol no-abort deviation in adjust_type).
    const symbolt *s =
      context.find_symbol(to_symbol_type(expr->type).symbol_name);
    if (s == nullptr || !s->is_type)
      return;
    type2tc resolved = ns.follow(expr->type);
    if (is_struct_type(resolved))
    {
      // Complete (pad) the followed type first so operand positions match
      // the final component list. Idempotent when already padded (S1).
      adjust_type(resolved);
      const struct_type2t &st = to_struct_type(resolved);
      std::vector<expr2tc> ops = to_constant_struct2t(expr).datatype_members;
      // Mirror the legacy already-padded heuristic: only insert padding
      // operands when the literal doesn't have them yet. Pad members are
      // recognised by the reserved `$` names (see the re-derivation in
      // adjust_type); inserting at component position i keeps the remaining
      // value operands aligned, exactly like the legacy insertion loop.
      if (ops.size() != st.members.size())
      {
        for (size_t i = 0; i < st.members.size(); i++)
        {
          const bool is_pad =
            is_padding_member_name(st.member_names[i].as_string());
          if (is_pad && i <= ops.size())
            ops.insert(ops.begin() + i, gen_zero(st.members[i]));
        }
      }
      // Rebuild only when the literal is structurally consistent; a residual
      // mismatch is left by-name for the exit invariant to flag.
      if (ops.size() == st.members.size())
        expr = constant_struct2tc(resolved, ops);
    }
  }
}

void python_adjust::adjust_type(type2tc &type)
{
  if (is_nil_type(type))
    return;

  if (is_symbol_type(type))
  {
    // Macro expansion only (legacy adjust_type `symbol.is_macro` arm): a
    // non-macro tag reference stays by-name and is followed at consumption
    // time — eagerly resolving it here would diverge from the legacy pass's
    // lazy sources (parity subtlety RV-adj6). Unlike the legacy pass this one
    // does not abort on an unknown *top-level* type symbol: the by-name type
    // is left untouched and, where it matters (member/index source, struct
    // literal), the post-adjust exit invariant flags it as an error instead.
    // An unknown tag buried inside an aggregate still aborts downstream
    // (add_padding's alignment() follows it via ns.follow, which asserts the
    // symbol exists) — the no-abort guarantee is top-level only.
    const symbolt *s = context.find_symbol(to_symbol_type(type).symbol_name);
    if (s != nullptr && s->is_type && s->is_macro)
    {
      type = s->get_type2();
      adjust_type(type);
    }
    return;
  }

  if (is_array_type(type))
  {
    // Adjust the (VLA) size expression and recurse into the element type
    // (legacy `is_array_like` arm; its vector_typet half has no analogue here
    // because the Python frontend never emits vector types). The nodes are
    // immutable — rebuild only when something changed so the pass stays inert
    // on complete types.
    const array_type2t &arr = to_array_type(type);
    type2tc subtype = arr.subtype;
    expr2tc size = arr.array_size;
    adjust_type(subtype);
    if (!is_nil_expr(size))
      adjust_expr(size);
    if (subtype != arr.subtype || size != arr.array_size)
      type = array_type2tc(subtype, size, arr.size_is_infinite);
    return;
  }

  if (is_struct_type(type) || is_union_type(type))
  {
    // Complete the aggregate (legacy struct/union arm): recurse the member
    // types, then insert alignment padding. Padding must reproduce the legacy
    // layout byte-for-byte (RV-adj5), so reuse add_padding itself through the
    // lossless type round-trip rather than reimplementing its alignment
    // arithmetic. On an already-completed type add_padding is a fixpoint
    // (asserted in the legacy pass), so this arm is idempotent and inert on
    // the live pipeline. IREP2 has no incomplete aggregates (they stay
    // symbol_type2t), so the legacy `!type.incomplete()` guard is not needed.
    auto members = is_struct_type(type) ? to_struct_type(type).members
                                        : to_union_type(type).members;
    bool members_changed = false;
    for (type2tc &member : members)
    {
      const type2tc before = member;
      adjust_type(member);
      members_changed |= member != before;
    }
    if (members_changed)
    {
      if (is_struct_type(type))
      {
        const struct_type2t &st = to_struct_type(type);
        type = struct_type2tc(
          members, st.member_names, st.member_pretty_names, st.name, st.packed);
      }
      else
      {
        const union_type2t &ut = to_union_type(type);
        type = union_type2tc(
          members, ut.member_names, ut.member_pretty_names, ut.name, ut.packed);
      }
    }

    typet legacy = migrate_type_back(type);
    // The #is_padding component flag does not survive the IREP2 round-trip,
    // and without it add_padding aligns an existing pad member as if it were
    // a regular field (padding.cpp:262 vs :276), double-padding the struct.
    // Re-derive it from the four reserved pad-member names add_padding
    // assigns: they all contain `$`, which cannot appear in a Python
    // identifier, so only add_padding's own members match. (The #bitfield/
    // #extint type flags are likewise dropped by the round-trip, but the
    // Python frontend never emits either, so only #is_padding needs
    // restoring.)
    for (auto &comp : to_struct_union_type(legacy).components())
      if (is_padding_member_name(comp.get_name().as_string()))
        comp.set_is_padding(true);
    add_padding(legacy, ns);
    type2tc padded = migrate_type(legacy);
    if (padded != type)
      type = padded;
    return;
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

void python_adjust::collect_unresolved_sources(
  const expr2tc &expr,
  std::vector<std::string> &out) const
{
  if (is_nil_expr(expr))
    return;

  // A member/index whose source type is still a symbol_type2t is unresolved:
  // resolve_source could not follow it to a concrete aggregate (e.g. it follows
  // to a non-aggregate scalar), so the strong construction invariant is unmet.
  // Each entry names the node kind and the by-name tag — together they are the
  // classification the B.5 resolution steps work from.
  if (is_member2t(expr) && is_symbol_type(to_member2t(expr).source_value->type))
  {
    const member2t &m = to_member2t(expr);
    out.push_back(
      "member `." + m.member.as_string() + "' over by-name source `" +
      to_symbol_type(m.source_value->type).symbol_name.as_string() + "'");
  }
  if (is_index2t(expr) && is_symbol_type(to_index2t(expr).source_value->type))
    out.push_back(
      "index over by-name source `" +
      to_symbol_type(to_index2t(expr).source_value->type)
        .symbol_name.as_string() +
      "'");
  // A constant_struct2t is the third relaxed construction assert (irep2_expr.h):
  // its own type may be a transient by-name symbol_type2t until the aggregate is
  // followed. Post-adjust it must be a resolved struct too.
  if (is_constant_struct2t(expr) && is_symbol_type(expr->type))
    out.push_back(
      "struct literal with by-name type `" +
      to_symbol_type(expr->type).symbol_name.as_string() + "'");
  // A resolved-struct literal must also be structurally consistent: the S2
  // completion only rebuilds when the operand count matches the component
  // list, so a count mismatch here means some other path retyped the literal
  // without inserting its padding operands — catch it before it reaches
  // migration/symex (constant_struct2t's constructor asserts only the type
  // kind, not the operand count).
  if (
    is_constant_struct2t(expr) && is_struct_type(expr->type) &&
    to_constant_struct2t(expr).datatype_members.size() !=
      to_struct_type(expr->type).members.size())
    out.push_back(
      "struct literal `" + to_struct_type(expr->type).name.as_string() +
      "' with " +
      std::to_string(to_constant_struct2t(expr).datatype_members.size()) +
      " operand(s) against " +
      std::to_string(to_struct_type(expr->type).members.size()) +
      " component(s)");

  expr->foreach_operand([this, &out](const expr2tc &e) {
    collect_unresolved_sources(e, out);
  });
}
