#include <python-frontend/python_adjust.h>

#include <clang-c-frontend/padding.h>
#include <irep2/irep2_utils.h>
#include <util/lang/c_types.h>
#include <util/message/message.h>
#include <util/irep/migrate.h>
#include <util/base/prefix.h>
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

  // Type symbols first, so symbolic-type resolution below always receives
  // fixed-up (macro-expanded, padded) tags — the same two-phase structure as
  // clang_c_adjust::adjust() (clang_c_adjust_expr.cpp:31-42). Scoped to
  // Python-mode symbols (RV-adj4): every converter-emitted tag is mode
  // "Python" (create_symbol, converter_util.cpp), while C/C++-header types
  // may contain bitfields whose #bitfield flag does not survive the IREP2
  // round-trip — re-padding those from the migrated view would write back a
  // corrupted layout. Inert on the live flag-on pipeline: clang_cpp_adjust
  // already completed every table type and adjust_type is a fixpoint on
  // complete types, so the write-back never fires until the flip makes this
  // pass the sole resolver.
  // Two further round-trip losses bound what the write-back below may carry
  // (both inert while the write-back never fires, both flip-era work): an
  // explicit "alignment" attribute is dropped by migrate_type, so an
  // over-aligned tag would be under-padded vs the legacy pass (the Python
  // frontend emits none); and legacy-only sub-ireps — most importantly the
  // "bases" list exception_typeid.cpp and base_type.cpp read for the
  // exception hierarchy — do not survive set_type(type2tc). See scope limit
  // (4) in the header.
  for (symbolt *symbol : symbol_list)
  {
    if (!symbol->is_type || symbol->mode != "Python")
      continue;

    const type2tc original = symbol->get_type2();
    type2tc t = original;
    adjust_type(t);
    if (t != original)
    {
      // Preserve the legacy-only "bases" sub-irep across the write-back:
      // set_type(type2tc) invalidates the legacy cache and the lazy
      // back-migration reconstructs only components/tag/packed, silently
      // dropping the inheritance list that exception-hierarchy consumers
      // (derive_exception_ids below, exception_typeid.cpp, base_type.cpp)
      // read. Re-attach it on the legacy view so both views stay coherent.
      // This discharges the "bases" half of scope limit (4) for the tags
      // this pre-pass rewrites; the remaining legacy-only attributes
      // (component access, #is_padding) stay cosmetic.
      const irept bases = symbol->get_type().find("bases");
      symbol->set_type(t);
      if (bases.is_not_nil())
      {
        typet patched = symbol->get_type();
        patched.set("bases", bases);
        symbol->set_type(std::move(patched));
      }
    }
  }

  bool error = false;
  for (symbolt *symbol : symbol_list)
  {
    if (symbol->is_type)
      continue;

    // Complete the non-type symbol's own type too (the legacy adjust_symbol
    // analogue, clang_c_adjust_expr.cpp:70-74) — scope limit (3) closing for
    // the by-name-alias shape (a pointer-to-code lambda variable stays with
    // the pinned call-rewrite; adjust_type has no pointer arm, legacy
    // parity). Python-mode only, same rationale as the type-symbol pre-pass;
    // write-back only on change keeps the live pipeline inert. Caveat for
    // the flip-era call rewrite: a resolved-alias code type written back
    // here carries no argument default_value (the attribute does not survive
    // the IREP2 round-trip) — default arguments must be sourced from the
    // function symbol, not from a variable's type.
    if (symbol->mode == "Python")
    {
      const type2tc t_original = symbol->get_type2();
      type2tc t = t_original;
      adjust_type(t);
      if (t != t_original)
        symbol->set_type(t);
    }

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
    // docs/roadmap/irep2-migration.md "S1 outcome" finding 2); S2's
    // aggregate-literal completion drained those, so a firing now means a node
    // shape the remaining S-steps (S3+) must resolve — the per-node detail
    // below is that work-list.
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

// Re-flag every pad member in the whole legacy type tree. add_padding recurses
// into component types (padding.cpp:71), so a nested aggregate that is already
// padded is re-padded unless *its* pad members carry #is_padding too — flagging
// only the top-level components leaves the inner ones looking like real fields.
void restore_padding_flags(typet &type)
{
  if (type.is_array())
  {
    restore_padding_flags(type.subtype());
    return;
  }

  if (!type.is_struct() && !type.is_union())
    return;

  for (auto &comp : to_struct_union_type(type).components())
  {
    if (is_padding_member_name(comp.get_name().as_string()))
      comp.set_is_padding(true);
    restore_padding_flags(comp.type());
  }
}

// Convert each argument to its declared parameter type, mirroring
// clang_c_adjust::adjust_function_call_arguments (clang_c_adjust_expr.cpp:1069).
// Callers decide which call forms reach it -- see adjust_expr.
//
// Cast only scalar/pointer kinds; an aggregate argument from an upstream typing
// bug keeps symex's own per-argument diagnostic rather than an unencodable
// typecast. Idempotent: a second pass sees `got == want` and rewrites nothing.
bool convert_call_arguments(const type2tc &callee, std::vector<expr2tc> &args)
{
  if (!is_code_type(callee))
    return false;

  const auto is_castable_kind = [](const type2tc &t) {
    return is_bv_type(t) || is_fixedbv_type(t) || is_floatbv_type(t) ||
           is_bool_type(t) || is_pointer_type(t);
  };

  const code_type2t &ct = to_code_type(callee);
  bool changed = false;
  for (size_t i = 0; i < args.size() && i < ct.arguments.size(); i++)
  {
    const type2tc &want = ct.arguments[i];
    const type2tc &got = args[i]->type;
    if (is_castable_kind(want) && is_castable_kind(got) && got != want)
    {
      args[i] = typecast2tc(want, args[i]);
      changed = true;
    }
  }
  return changed;
}

// Insert a gen_zero operand at each reserved padding-member position so the
// literal's operand list matches the struct's component list, exactly as the
// legacy adjust_struct insertion loop does. Idempotent when already padded.
std::vector<expr2tc>
pad_struct_operands(const struct_type2t &st, std::vector<expr2tc> ops)
{
  for (size_t i = 0; i < st.members.size(); i++)
    if (
      i <= ops.size() && is_padding_member_name(st.member_names[i].as_string()))
      ops.insert(ops.begin() + i, gen_zero(st.members[i]));
  return ops;
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
    bool rebuild = false;

    if (is_pointer_type(source->type))
    {
      // clang_c_adjust::adjust_member wraps a pointer base in a dereference
      // (clang_c_adjust_expr.cpp:307-313), so `p.field` becomes `p->field`.
      // The converter emits the unwrapped form for a class attribute reached
      // through an instance pointer (`cur.nxt` where `cur` is a `Node`
      // parameter), and without the wrap the member's source stays a pointer:
      // symex then reads a member off a pointer value and aborts
      // ("to_pointer_type() called on type whose type_id is struct"), and the
      // expression printer falls back to dumping the raw irep.
      source = dereference2tc(to_pointer_type(source->type).subtype, source);
      rebuild = true;
    }

    // The wrapped source carries the pointee, which is the same transient
    // symbol_type2t a plain instance source would be -- resolve either shape.
    if (resolve_source(source))
      rebuild = true;

    if (rebuild)
      expr = member2tc(m.type, source, m.member);
  }
  else if (is_index2t(expr))
  {
    const index2t &i = to_index2t(expr);
    // clang_c_adjust::adjust_index casts the index to index_type() before using
    // it (clang_c_adjust_expr.cpp:591). Without it an index of a different width
    // or signedness reaches the element computation unconverted — e.g.
    // `float_buf[obj->float_idx]` where legacy emits
    // `float_buf[(signed long int)obj->float_idx]` — which changes the value read
    // and can flip a verdict.
    expr2tc idx = i.index;
    if (idx->type != index_type2())
      idx = typecast2tc(index_type2(), idx);

    if (is_pointer_type(i.source_value->type))
    {
      // clang_c_adjust::adjust_index rewrites p[i] -> *(p+i) when the base is a
      // pointer (a Python string / decayed-array source). Requires the
      // array→pointer decay arm below so the pointer source actually holds a
      // pointer value at symex rename, not a bare array.
      expr = dereference2tc(
        i.type, add2tc(i.source_value->type, i.source_value, idx));
    }
    else
    {
      expr2tc source = i.source_value;
      const bool source_resolved = resolve_source(source);
      if (source_resolved || idx != i.index)
        expr = index2tc(i.type, source, idx);
    }
  }
  else if (is_dereference2t(expr) && is_empty_type(expr->type))
  {
    // A pointer dereference whose result type the converter left empty -- a
    // Python element access `s[i]` over a char*-like source (chr()'s result is
    // the canonical case). clang_cpp_adjust resolves the read type to the
    // pointee; do the same so symex does not get_width() an empty deref target
    // (the S3 symbolic_type_excp root, docs/roadmap/scope-v1k-adjuster
    // round-4). Only when the pointee is non-empty -- a void*-like empty
    // pointee is left for the exit invariant, exactly as clang leaves a void
    // deref empty. An array operand (clang's `*a` -> `a[0]` rewrite) does not
    // occur on the Python path (subscripts lower to index2t). dereference2t is
    // immutable, rebuild.
    const dereference2t &d = to_dereference2t(expr);
    if (is_pointer_type(d.value->type))
    {
      const type2tc &pointee = to_pointer_type(d.value->type).subtype;
      if (!is_empty_type(pointee))
        expr = dereference2tc(pointee, d.value);
    }
  }
  else if (is_if2t(expr) && !is_bool_type(to_if2t(expr).cond->type))
  {
    // A ternary whose condition is not boolean -- a non-boolean short-circuit
    // `and`/`or` select builds `cond ? a : b` with the raw integer operand as
    // the condition (get_truthy_condition returns a non-list value unchanged,
    // e.g. `len(s)` in `len(s) or len(t)`). clang_c_adjust::adjust_if casts the
    // condition to bool (gen_typecast(ns, op0, bool_type())); mirror it so
    // goto_sideeffects' is_boolean() check on the lowered IF condition holds
    // (otherwise "first argument of `if' must be boolean"). Its sibling half --
    // converting a branch whose type differs from the result type -- is not
    // mirrored: migrate_expr's ternary arm coerces every branch whose type
    // *kind* diverges (migrate.cpp:1001, exactly what if2t's assert demands),
    // and the residual same-kind/different-width case is unobserved on the
    // Python path (0 firings across 40 ternary-bearing tests), so the mirror
    // would be dead instrumentation today. if2t is immutable.
    const if2t &i = to_if2t(expr);
    expr = if2tc(
      i.type,
      typecast2tc(get_bool_type(), i.cond),
      i.true_value,
      i.false_value);
  }
  else if (is_not2t(expr) && !is_bool_type(to_not2t(expr).value->type))
  {
    // clang_c_adjust::adjust_expr_unary_boolean casts `not`'s operand to bool
    // (clang_c_adjust_expr.cpp:1530-1538). Python's `not x` over a non-boolean
    // value -- `not (x and True)` where `x` is None, so the short-circuit select
    // has pointer type -- otherwise reaches the SMT layer as a negation of a
    // non-boolean sort and trips bitwuzla's mk_not assert. not2t is immutable.
    expr = not2tc(typecast2tc(get_bool_type(), to_not2t(expr).value));
  }
  else if (
    is_typecast2t(expr) && is_pointer_type(expr->type) &&
    is_array_type(to_typecast2t(expr).from->type))
  {
    // A cast of an array value to a pointer decays: `(char *)arr` is
    // `&arr[0]`. clang never builds the raw cast -- every conversion it emits
    // goes through c_typecastt::do_typecast, whose array case decays
    // (c_typecast.cpp:926-944, the expr2tc overload this reimplements rather
    // than calls: the pass deliberately keeps legacy c_typecastt off the
    // IREP2-native path). Restricted to a pointer destination, the only shape
    // migrate_expr's coercion produces here; do_typecast decays for any
    // destination. On the Python path the raw cast is synthesised
    // by migrate_expr's ternary arm, which coerces a branch whose type id
    // diverges from the result type (migrate.cpp:1001): `s = "" if b else
    // "foo"` builds a char*-typed if over two array literals, so both branches
    // arrive here as `(char *){ ... }`. The SMT layer then rejects the
    // pointer-typed array constant ("Unexpected type in int/ptr typecast").
    // Idempotent: the rewritten node is an address_of, not an array.
    const typecast2t &t = to_typecast2t(expr);
    const type2tc &elem = to_array_type(t.from->type).subtype;
    expr2tc decayed =
      address_of2tc(elem, index2tc(elem, t.from, gen_zero(index_type2())));
    expr = (ns.follow(decayed->type) == ns.follow(expr->type))
             ? decayed
             : typecast2tc(expr->type, decayed);
  }
  else if (
    is_address_of2t(expr) && is_array_type(to_address_of2t(expr).ptr_obj->type))
  {
    // `&array` decays to `&array[0]`, exactly as clang_c_adjust::adjust_address_of
    // does (clang_c_adjust_expr.cpp:743-754). This is the node-level counterpart
    // of the assignment-seam decay below: the operand need not be near an
    // assignment at all -- the OM raise sites build a struct literal
    // `{ .message = &"math domain error" }` whose member is a `char*`, so
    // without the decay the literal carries a `char(*)[N]` and the member type
    // silently disagrees with its initialiser. Idempotent: the rewritten operand
    // is an index2t of element type, so the arm cannot re-fire.
    const address_of2t &a = to_address_of2t(expr);
    const type2tc &elem = to_array_type(a.ptr_obj->type).subtype;
    expr =
      address_of2tc(elem, index2tc(elem, a.ptr_obj, gen_zero(index_type2())));
  }
  else if (
    is_code_assign2t(expr) &&
    is_pointer_type(to_code_assign2t(expr).target->type) &&
    is_array_type(to_code_assign2t(expr).source->type))
  {
    // Array→pointer decay at the assignment seam: a `char*` target assigned a
    // bare array value (a Python string literal, e.g. `word = ""` where `""` is
    // a constant_array) must decay to `&array[0]`, exactly as clang_c_adjust
    // lowers it (`ASSIGN word = &{0}[0]`). Without it the pointer variable
    // carries an array value and any pointer use of it (indexing, arithmetic)
    // trips a pointer-vs-array mismatch at symex rename (irep2_cast_error in
    // fixup_renamed_type). code_assign2t is immutable, rebuild.
    const code_assign2t &a = to_code_assign2t(expr);
    const type2tc &elem = to_array_type(a.source->type).subtype;
    // address_of2t's type is pointer-to-<subtype>, so pass the target's pointee
    // (not the full pointer type) — the rebuilt value is then exactly
    // a.target->type, matching clang's c_typecast (address_of2tc(ptr.subtype,
    // index)), not pointer(pointer(elem)).
    const type2tc &pointee = to_pointer_type(a.target->type).subtype;
    expr2tc decayed =
      address_of2tc(pointee, index2tc(elem, a.source, gen_zero(index_type2())));
    expr = code_assign2tc(a.target, decayed, a.location);
  }
  else if (
    is_code_assign2t(expr) &&
    is_pointer_type(to_code_assign2t(expr).target->type) &&
    is_struct_type(ns.follow(to_code_assign2t(expr).source->type)))
  {
    // The struct sibling of the decay above: a pointer target assigned an
    // aggregate *value* takes its address. c_typecastt::implicit_typecast_
    // followed does this for a struct or union source (c_typecast.cpp:729-740,
    // the `address_of_exprt base_ptr` arm). The Python converter binds an
    // instance parameter this way -- `cur = head` where `head` is a
    // `pointer→tag-Node` parameter lowers to `cur = *head`, a struct value --
    // and legacy emits `cur = &(*head)`. Without the address-of, symex reads a
    // struct where the pointer's type says pointer and aborts
    // ("to_pointer_type() called on type whose type_id is struct").
    // Idempotent: the rebuilt source is an address_of, not a struct.
    const code_assign2t &a = to_code_assign2t(expr);
    const type2tc &pointee = to_pointer_type(a.target->type).subtype;
    expr =
      code_assign2tc(a.target, address_of2tc(pointee, a.source), a.location);
  }
  else if (
    is_code_assign2t(expr) && is_sideeffect2t(to_code_assign2t(expr).source) &&
    to_sideeffect2t(to_code_assign2t(expr).source).kind ==
      sideeffect2t::allockind::function_call &&
    to_code_assign2t(expr).source->type != to_code_assign2t(expr).target->type)
  {
    // Convert a call result to the target's type, the one shape of
    // clang_c_adjust::adjust_assign's gen_typecast that is safe to mirror
    // alone. `length = len(xs)` binds an `unsigned long` model return to a
    // `signed long` variable; without the conversion convert_assign's
    // call-valued-rhs special case (goto_convert.cpp) hands the lhs straight to
    // do_function_call, so no temporary and no cast is emitted and the signed
    // variable holds an unsigned value -- a later `i < length` then reaches the
    // solver as a lessthan2t over mismatched operand kinds.
    //
    // The general assignment conversion stays parked (see the
    // assignment-conversion trap in docs/roadmap/scope-v1k-adjuster.md): it is
    // only sound coupled with operand-level arithmetic reconciliation, and
    // shipping it alone masks a real bug in neural-net_fail. That coupling is
    // about reconciling a *binary operation's* operands on the right-hand side,
    // which a call source has none of -- so this shape carries none of that
    // risk, and neural-net_fail was re-checked with this arm in place.
    const code_assign2t &a = to_code_assign2t(expr);
    expr = code_assign2tc(
      a.target, typecast2tc(a.target->type, a.source), a.location);
  }
  else if (
    is_code_ifthenelse2t(expr) &&
    !is_bool_type(to_code_ifthenelse2t(expr).cond->type))
  {
    // Branch/loop conditions must be boolean before the solver sees them. Python
    // writes `if x:` on a plain int, and the converter keeps the raw signedbv;
    // clang_c_adjust casts it (adjust_ifthenelse/adjust_while/adjust_for all call
    // gen_typecast_bool). Without the cast the guard reaches the SMT layer as a
    // bitvector where a Boolean is required -- bitwuzla rejects it with "term
    // with unexpected sort at index 0". This is the statement-level counterpart
    // of the if2t (ternary) arm above.
    const code_ifthenelse2t &i = to_code_ifthenelse2t(expr);
    expr = code_ifthenelse2tc(
      typecast2tc(get_bool_type(), i.cond),
      i.then_case,
      i.else_case,
      i.location);
  }
  else if (
    is_code_while2t(expr) && !is_bool_type(to_code_while2t(expr).cond->type))
  {
    const code_while2t &w = to_code_while2t(expr);
    expr = code_while2tc(
      typecast2tc(get_bool_type(), w.cond),
      w.body,
      w.location,
      w.pragma_unroll_count);
  }
  else if (
    is_code_dowhile2t(expr) &&
    !is_bool_type(to_code_dowhile2t(expr).cond->type))
  {
    const code_dowhile2t &d = to_code_dowhile2t(expr);
    expr = code_dowhile2tc(
      typecast2tc(get_bool_type(), d.cond),
      d.body,
      d.location,
      d.pragma_unroll_count);
  }
  else if (
    is_code_for2t(expr) && !is_nil_expr(to_code_for2t(expr).cond) &&
    !is_bool_type(to_code_for2t(expr).cond->type))
  {
    const code_for2t &f = to_code_for2t(expr);
    expr = code_for2tc(
      f.init,
      typecast2tc(get_bool_type(), f.cond),
      f.iter,
      f.body,
      f.location,
      f.pragma_unroll_count);
  }
  else if (
    is_code_return2t(expr) && !is_nil_expr(to_code_return2t(expr).operand) &&
    is_code_type(to_code_return2t(expr).operand->type))
  {
    // Function→pointer decay at the return seam (C11 6.3.2.1p4): a closure
    // factory (`def make(k): def mul(x): ...; return mul`) returns a bare
    // code-typed designator, but the caller stores it in a function pointer.
    // clang_c_adjust decays every code-typed symbol reference to `&f`
    // (adjust_symbol_expr, "sugar for &f"); mirror it at the one seam Python
    // reaches it from. Without it symex sees `typecast(mul, void(*)())` and
    // aborts at SMT encoding ("Unexpected type in int/ptr typecast"), and the
    // indirect call has no resolvable target.
    const code_return2t &r = to_code_return2t(expr);
    expr =
      code_return2tc(address_of2tc(r.operand->type, r.operand), r.location);
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
    // (raise IndexError(...) et al., docs/roadmap/irep2-migration.md "S1
    // outcome" finding 2): their operands were already padded by the legacy
    // pass, so only the retype fires; the padding-operand insertion below
    // completes a converter-built literal once the flip makes this pass the
    // sole resolver.
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
      // Mirror the legacy already-padded heuristic: only insert padding
      // operands when the literal doesn't have them yet. pad_struct_operands
      // is not idempotent, so the size guard must gate the call.
      std::vector<expr2tc> ops = to_constant_struct2t(expr).datatype_members;
      if (ops.size() != st.members.size())
        ops = pad_struct_operands(st, ops);
      // Rebuild only when the literal is structurally consistent; a residual
      // mismatch is left by-name for the exit invariant to flag.
      if (ops.size() == st.members.size())
        expr = constant_struct2tc(resolved, ops);
    }
  }
  else if (
    is_constant_struct2t(expr) && is_struct_type(expr->type) &&
    to_constant_struct2t(expr).datatype_members.size() !=
      to_struct_type(expr->type).members.size())
  {
    // A literal already retyped to a resolved struct but left with fewer
    // operands than components — the converter built an Optional/union literal
    // (e.g. `int | None`: `{ is_none, anon_pad$, value }`) without its padding
    // operand, and no legacy adjust_struct ran to insert it. Pad it the same
    // way as the by-name S2 arm above; the type is already resolved so no
    // follow is needed. A residual mismatch is left for the exit invariant.
    const struct_type2t &st = to_struct_type(expr->type);
    std::vector<expr2tc> ops =
      pad_struct_operands(st, to_constant_struct2t(expr).datatype_members);
    if (ops.size() == st.members.size())
      expr = constant_struct2tc(expr->type, ops);
  }
  else if (is_code_function_call2t(expr))
  {
    // Statement-form call through a lambda/def-alias variable.
    const code_function_call2t &c = to_code_function_call2t(expr);
    expr2tc fn = c.function;
    std::vector<expr2tc> args = c.operands;
    if (wrap_function_pointer_callee(fn, args))
      expr = code_function_call2tc(c.ret, fn, args, c.location);
  }
  else if (
    is_sideeffect2t(expr) &&
    to_sideeffect2t(expr).kind == sideeffect2t::allockind::function_call)
  {
    // Expression-form call (e.g. `assert f(3) == 6`): the callee is the
    // sideeffect operand.
    const sideeffect2t &s = to_sideeffect2t(expr);

    // A C-library math call lowers to its SMT intrinsic instead of executing
    // the model, mirroring clang_c_adjust (`sqrt` at
    // clang_c_adjust_expr.cpp:1414-1423, `fabs` at :1239-1245). math.sqrt /
    // math.fabs build calls to `c:@F@sqrt` / `c:@F@fabs`
    // (python_math::handle_sqrt, build_unary_c_math_call); without the lowering
    // the hop-off runs the library model -- for sqrt that yields NaN, so
    // `math.sqrt(9) == 3.0` reports a spurious violation.
    //
    // These two are the whole intersection of the names Python emits as
    // `c:@F@` calls with the names clang_c_adjust lowers (its other eleven --
    // finite/fma/huge_val/inf/isfinite/isinf/isnan/isnormal/nan/nearbyint/
    // signbit -- are never reached as calls from this frontend), so a further
    // arm here would be dead instrumentation.
    //
    // The legacy guard matches the symbol's *base* name and excludes `py:` user
    // functions; symbol2t carries only the full identifier, so take the segment
    // after the last '@'. ieee_sqrt's rounding mode matches migrate_expr's
    // default for a legacy node with no explicit mode (migrate.cpp:1437).
    if (is_symbol2t(s.operand) && s.arguments.size() == 1)
    {
      const std::string id = to_symbol2t(s.operand).thename.as_string();
      const std::string base = id.substr(id.find_last_of('@') + 1);
      const auto is_float_variant = [&base](const std::string &n) {
        return base == n || base == n + "f" || base == n + "d" ||
               base == n + "l";
      };
      if (!has_prefix(id, "py:"))
      {
        if (is_float_variant("sqrt"))
        {
          expr = ieee_sqrt2tc(
            s.type,
            s.arguments[0],
            symbol2tc(get_int32_type(), "c:@__ESBMC_rounding_mode"));
          return;
        }
        if (is_float_variant("fabs"))
        {
          expr = abs2tc(s.type, s.arguments[0]);
          return;
        }
      }
    }

    expr2tc fn = s.operand;
    std::vector<expr2tc> args = s.arguments;
    bool changed = wrap_function_pointer_callee(fn, args);

    // Argument conversion belongs to this form only. Legacy reaches
    // adjust_function_call_arguments exclusively via
    // adjust_side_effect_function_call; its statement-form arm
    // (clang_c_adjust_code.cpp, `statement == "function_call"`) adjusts index
    // expressions and nothing else, so a statement-form call keeps its
    // arguments verbatim. Converting both forms is a measured parity
    // regression -- it casts the list-model calls (`list_push(result,
    // (void *)(&elem), ...)`) legacy leaves alone, trading one closed diff for
    // roughly ten new ones.
    changed |= convert_call_arguments(fn->type, args);
    if (changed)
      expr = sideeffect2tc(s.type, fn, s.size, args, s.alloctype, s.kind);
  }
  else if (is_code_cpp_throw2t(expr))
  {
    // Flip blocker #1 (docs/roadmap/irep2-migration.md, "Flip-probe census"):
    // the exception-id chain is derived only by clang_cpp_adjust today
    // (adjust_side_effect_throw); once that hop is gone, every operand-carrying
    // THROW reaches remove_exceptions with an empty exception_list and crashes
    // its unguarded front(). Complete an empty list here from the operand's
    // class type. A list the legacy pass already filled is left untouched, so
    // this arm is inert until the flip; a bare re-raise (nil operand) keeps its
    // empty list, as legacy does. The operand was already recursed above, so
    // under S2 its type may be the resolved struct rather than the by-name tag
    // — both derive the same chain.
    const code_cpp_throw2t &t = to_code_cpp_throw2t(expr);
    if (t.exception_list.empty() && !is_nil_expr(t.operand))
    {
      const std::vector<irep_idt> ids = derive_exception_ids(t.operand->type);
      if (!ids.empty())
        expr = code_cpp_throw2tc(t.operand, ids, t.location);
    }
  }
}

bool python_adjust::wrap_function_pointer_callee(
  expr2tc &fn,
  std::vector<expr2tc> &args)
{
  // A lambda/def-alias call (`op = lambda ...; op(3)`): the callee symbol's
  // table type is pointer-to-code, but goto-convert wants a code-typed
  // callee. Re-type it from the table, dereference onto the code type, and
  // cast each argument to its declared parameter type — the legacy
  // adjust_symbol + implicit-deref + adjust_function_call_arguments trio.
  // Inert on the default pipeline (legacy rewrites these calls before
  // migration, so the callee already arrives as a dereference).
  if (is_nil_expr(fn))
    return false;

  const symbolt *fs =
    is_symbol2t(fn) ? context.find_symbol(to_symbol2t(fn).thename) : nullptr;

  // Any other pointer-to-code callee -- a lambda read back out of a container,
  // `{'+': lambda: 1.0}[x]()`, whose callee is a typecast of a member read, not
  // a symbol. clang_c_adjust::adjust_side_effect_function_call dereferences
  // *any* pointer-typed callee (clang_c_adjust_expr.cpp, the implicit-deref
  // arm); without it goto-convert calls through the pointer value itself and
  // the result is read under the wrong signature. This path fixes the callee
  // only -- argument conversion is the call site's job in adjust_expr.
  if (fs == nullptr || !is_pointer_type(fs->get_type2()))
  {
    if (!is_pointer_type(fn->type))
      return false;
    const type2tc &pointee = to_pointer_type(fn->type).subtype;
    if (!is_code_type(pointee))
      return false;
    fn = dereference2tc(pointee, fn);
    return true;
  }

  // Python points directly at the code type (no typedefs to follow).
  const type2tc &pointee = to_pointer_type(fs->get_type2()).subtype;
  if (!is_code_type(pointee))
    return false;
  const irep_idt &name = to_symbol2t(fn).thename;

  convert_call_arguments(pointee, args);

  // Build the dereference over the code type — goto-convert's dispatch
  // wants a code-typed callee.
  fn = dereference2tc(pointee, symbol2tc(fs->get_type2(), name));
  return true;
}

std::vector<irep_idt>
python_adjust::derive_exception_ids(const type2tc &type) const
{
  std::vector<irep_idt> ids;
  derive_exception_ids_rec(type, "", ids);
  return ids;
}

void python_adjust::derive_exception_ids_rec(
  const type2tc &type,
  const std::string &suffix,
  std::vector<irep_idt> &ids) const
{
  // Mirror clang_cpp_adjust::convert_exception_id for the shapes the Python
  // frontend emits (remove_exceptions' register_chain builds the transitive
  // hierarchy from these one-level chains). A pointer operand is real: the
  // untypeable-raise fallback types the operand any_type() = pointer(empty)
  // (python_exception_handler get_raise_statement), which legacy derives as
  // "void_ptr". The trailing never-empty fallback mirrors legacy's — callers
  // (remove_exceptions) dereference front(), so an unknown shape must yield
  // a synthetic id that simply never matches a real throw, not an empty
  // list. (Legacy also appends a `#cpp_type` id when present; that attribute
  // does not survive migration and Python types never carry it.)
  if (is_pointer_type(type))
  {
    const type2tc &sub = to_pointer_type(type).subtype;
    if (is_empty_type(sub))
      ids.emplace_back("void_ptr" + suffix);
    else
      derive_exception_ids_rec(sub, "_ptr" + suffix, ids);
    return;
  }

  std::string bare;
  if (is_symbol_type(type))
  {
    const std::string id = to_symbol_type(type).symbol_name.as_string();
    bare = has_prefix(id, "tag-") ? id.substr(4) : id;
  }
  else if (is_struct_type(type))
    // migrate_type stores the legacy `tag` attribute — the bare class name.
    bare = to_struct_type(type).name.as_string();

  if (!bare.empty())
  {
    ids.emplace_back(bare + suffix);
    // Direct bases, declaration order, one level — exactly the legacy
    // derivation. The "bases" list lives only on the legacy view of the tag
    // symbol (the W3 attribute-carriage gap); the type-symbol pre-pass
    // preserves it across its write-back precisely so this read stays valid.
    const symbolt *tag = context.find_symbol("tag-" + bare);
    if (tag != nullptr && tag->is_type && tag->get_type().is_struct())
    {
      const irept &bases = tag->get_type().find("bases");
      for (const auto &b : bases.get_sub())
      {
        const std::string bid = b.id().as_string();
        ids.emplace_back(
          (has_prefix(bid, "tag-") ? bid.substr(4) : bid) + suffix);
      }
    }
  }

  if (ids.empty())
    ids.emplace_back(get_type_id(type) + suffix);
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

  if (is_code_type(type))
  {
    // Pad any struct/union embedded in the function signature (argument and
    // return types), so a function argument's Optional/union type matches the
    // padded value literal at the call site — otherwise symex_function's
    // base_type_eq rejects a padded argument against an unpadded parameter
    // ("argument type mismatch: got struct, expected struct"). Inert on a
    // signature that carries only scalars.
    const code_type2t &ct = to_code_type(type);
    std::vector<type2tc> args = ct.arguments;
    type2tc ret = ct.ret_type;
    bool changed = false;
    for (type2tc &a : args)
    {
      const type2tc before = a;
      adjust_type(a);
      changed |= a != before;
    }
    const type2tc ret_before = ret;
    adjust_type(ret);
    changed |= ret != ret_before;
    if (changed)
      type = code_type2tc(args, ret, ct.argument_names, ct.ellipsis);
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
    // restoring.) The walk must be recursive: add_padding pads component types
    // before the enclosing one, so an already-padded nested aggregate (an
    // `int | None` attribute inside its class struct) is re-padded unless its
    // own pad members are flagged too.
    restore_padding_flags(legacy);
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
  // A pointer source is transient too (the index arm rewrites `p[i]` to
  // `*(p+i)`); one surviving here means the rewrite was skipped, so symex would
  // see an index over a pointer — flag it before it escapes.
  if (is_index2t(expr) && is_pointer_type(to_index2t(expr).source_value->type))
    out.push_back("index over unresolved pointer source");
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

  expr->foreach_operand(
    [this, &out](const expr2tc &e) { collect_unresolved_sources(e, out); });
}
