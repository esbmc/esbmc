#include <goto-programs/goto_check_uninit_vars.h>

#include <map>
#include <set>
#include <vector>
#include <util/c_types.h>
#include <util/migrate.h>
#include <util/prefix.h>
#include <util/symbol_generator.h>

namespace
{
/// Maximum element count of a fixed-size array we will shadow per-element.
/// Above this threshold the parallel `bool[N]` shadow (and the `array-of`
/// initialiser / whole-flip writes that go with it) inflates the SMT
/// encoding more than the precision is worth, so the array is left
/// untracked — same observable behaviour as before PR #4507 for that
/// variable. Tuned to cover typical stack/firmware buffers (a few KB) while
/// avoiding blow-up on programs like `int buf[100000];`.
constexpr uint64_t kMaxShadowedArraySize = 4096;

/// Element types tracked for uninitialised-read detection (CWE-457). Pointers
/// are reported under CWE-908; aggregates (struct/union) are out of scope.
bool is_scalar_element_id(const irep_idt &id)
{
  return id == "bool" || id == "signedbv" || id == "unsignedbv" ||
         id == "fixedbv" || id == "floatbv";
}

/// Build the shadow type matching a tracked user local. Scalars get a single
/// `bool`; fixed-size 1-D arrays of scalars whose constant length is at
/// most `kMaxShadowedArraySize` get a parallel `bool[N]`. Returns nil for
/// anything else: pointers, VLAs (non-constant length), infinite/incomplete
/// arrays, multi-dim arrays, aggregates, and oversize arrays.
type2tc shadow_type_for(const typet &user_type)
{
  if (is_scalar_element_id(user_type.id()))
    return get_bool_type();

  if (user_type.id() != "array")
    return type2tc();

  if (!is_scalar_element_id(user_type.subtype().id()))
    return type2tc();

  type2tc migrated = migrate_type(user_type);
  const array_type2t &arr = to_array_type(migrated);
  if (arr.size_is_infinite || is_nil_expr(arr.array_size))
    return type2tc();
  // VLAs reach here with a runtime-expression size; require a constant.
  if (!is_constant_int2t(arr.array_size))
    return type2tc();
  const BigInt &n = to_constant_int2t(arr.array_size).value;
  if (
    n.is_negative() || n.is_zero() ||
    n.compare(static_cast<unsigned long long>(kMaxShadowedArraySize)) > 0)
    return type2tc();
  return array_type2tc(get_bool_type(), arr.array_size, false);
}

/// Tracked iff automatic storage, lvalue, not internal/return, and the type
/// has a shadow representation.
type2tc tracked_shadow_type(const symbolt &s)
{
  if (s.static_lifetime || s.is_extern || !s.lvalue)
    return type2tc();
  if (has_prefix(s.name, "return_value$"))
    return type2tc();
  if (has_prefix(s.id, "__ESBMC_") || has_prefix(s.name, "__ESBMC_"))
    return type2tc();
  return shadow_type_for(s.type);
}

/// Record describing one shadow variable, looked up at emit time by sid.
struct shadow_record
{
  type2tc type; // bool, or bool[N]
  std::string user_name;
};

/// Build the shadow-initialiser/flip RHS for a tracked variable: a plain
/// boolean for scalars, or `constant_array_of(value)` for arrays.
expr2tc shadow_const(const type2tc &shadow_t, const expr2tc &value)
{
  return is_array_type(shadow_t)
           ? expr2tc(constant_array_of2tc(shadow_t, value))
           : value;
}

/// Walk an lvalue chain and return its root symbol name, or empty if the
/// chain bottoms out in something other than a symbol (e.g. a dereference).
irep_idt lvalue_root(const expr2tc &e)
{
  const expr2tc *cur = &e;
  while (true)
  {
    if (is_nil_expr(*cur))
      return irep_idt();
    if (is_symbol2t(*cur))
      return to_symbol2t(*cur).thename;
    if (is_index2t(*cur))
    {
      cur = &to_index2t(*cur).source_value;
      continue;
    }
    if (is_member2t(*cur))
    {
      cur = &to_member2t(*cur).source_value;
      continue;
    }
    return irep_idt();
  }
}

/// Per-instruction collection of reads/writes against tracked shadows.
struct collected
{
  std::set<irep_idt> scalar_reads; // shadow ids read as whole scalars
  std::set<irep_idt> whole_flips;  // shadow ids whose whole shadow flips true
  // (shadow id, index expression) pairs read; one ASSERT per entry.
  std::vector<std::pair<irep_idt, expr2tc>> indexed_reads;
};

/// Look up the shadow id for `name`, or empty if `name` is not tracked.
irep_idt shadow_of(irep_idt name, const std::map<irep_idt, irep_idt> &shadows)
{
  auto it = shadows.find(name);
  return it == shadows.end() ? irep_idt() : it->second;
}

/// Walk an expression collecting reads of tracked shadows.
///
/// `lvalue_ctx` is true while inside an l-value chain (operand of an
/// `address_of2t`, or recursion through `member2t`/`index2t.source_value`
/// rooted at an address-of). In that mode the root symbol is not a value-read;
/// embedded sub-expressions (for example `i` in `a[i]`) still are.
void collect(
  const expr2tc &expr,
  bool lvalue_ctx,
  const std::map<irep_idt, irep_idt> &shadows,
  const std::map<irep_idt, shadow_record> &by_sid,
  collected &out)
{
  if (is_nil_expr(expr))
    return;

  if (is_address_of2t(expr))
  {
    const expr2tc &target = to_address_of2t(expr).ptr_obj;
    irep_idt sid = shadow_of(lvalue_root(target), shadows);
    if (!sid.empty())
      out.whole_flips.insert(sid);
    // Recurse into target as an l-value so embedded reads (e.g. `i` in
    // `&a[i]`) are still checked, but the base symbol itself is not.
    collect(target, true, shadows, by_sid, out);
    return;
  }

  if (is_symbol2t(expr))
  {
    if (lvalue_ctx)
      return; // base symbol of an l-value chain; not a value-read
    irep_idt sid = shadow_of(to_symbol2t(expr).thename, shadows);
    if (sid.empty())
      return;
    if (is_array_type(by_sid.at(sid).type))
      return; // bare array name as r-value (rare/degenerate); skip
    out.scalar_reads.insert(sid);
    return;
  }

  if (is_index2t(expr))
  {
    const index2t &i = to_index2t(expr);
    if (lvalue_ctx)
    {
      // Inside an l-value chain: base remains lvalue, index is a read.
      collect(i.source_value, true, shadows, by_sid, out);
      collect(i.index, false, shadows, by_sid, out);
      return;
    }
    // R-value `a[i]`: if `a` resolves to a tracked array, record the
    // (shadow, index) pair and treat the index itself as a read. Do not
    // recurse into the base symbol — there is no scalar shadow for `a`.
    irep_idt sid = shadow_of(lvalue_root(i.source_value), shadows);
    if (!sid.empty() && is_array_type(by_sid.at(sid).type))
    {
      out.indexed_reads.emplace_back(sid, i.index);
      collect(i.index, false, shadows, by_sid, out);
      return;
    }
    // Not a tracked array: fall through to default operand walker.
  }

  if (lvalue_ctx && is_member2t(expr))
  {
    collect(to_member2t(expr).source_value, true, shadows, by_sid, out);
    return; // member name is a literal
  }

  // In l-value mode, a `dereference2t` exits the l-value context: the
  // pointer operand is itself a value-read.
  if (lvalue_ctx && is_dereference2t(expr))
  {
    collect(to_dereference2t(expr).value, false, shadows, by_sid, out);
    return;
  }

  // Default: preserve the current context for unknown wrappers and walk all
  // operands. Preserving avoids treating a base lvalue symbol as a value-read
  // — a false-positive risk we cannot tolerate as a formal verifier.
  expr->foreach_operand(
    [&](const expr2tc &e) { collect(e, lvalue_ctx, shadows, by_sid, out); });
}

/// Direct write target for ASSIGN/FUNCTION_CALL.ret: either a bare tracked
/// symbol (scalar whole-flip, or array whole-flip on `a = …`) or a bare
/// tracked-array element `a[i]` (per-element flip). Anything else returns
/// empty.
struct write_target
{
  irep_idt root; // user name, empty if not a direct write to a tracked
  expr2tc index; // non-nil iff root names an `a[i]` write
};

write_target direct_write_target(
  const expr2tc &lhs,
  const std::map<irep_idt, irep_idt> &shadows,
  const std::map<irep_idt, shadow_record> &by_sid)
{
  if (is_nil_expr(lhs))
    return {};
  if (is_symbol2t(lhs))
    return {to_symbol2t(lhs).thename, expr2tc()};
  if (is_index2t(lhs))
  {
    const index2t &i = to_index2t(lhs);
    if (!is_symbol2t(i.source_value))
      return {};
    irep_idt name = to_symbol2t(i.source_value).thename;
    irep_idt sid = shadow_of(name, shadows);
    if (sid.empty() || !is_array_type(by_sid.at(sid).type))
      return {};
    return {name, i.index};
  }
  return {};
}
} // namespace

bool goto_check_uninit_vars::runOnFunction(
  std::pair<const irep_idt, goto_functiont> &F)
{
  if (!F.second.body_available)
    return false;

  goto_programt &body = F.second.body;
  std::map<irep_idt, irep_idt> shadows;     // user-name -> shadow_id
  std::map<irep_idt, shadow_record> by_sid; // shadow_id -> record
  // Prefix uppercase so `tracked_shadow_type`'s "__ESBMC_" filter at the next
  // pass invocation rejects our own shadows — defence in depth against
  // self-instrumentation on a re-run.
  symbol_generator gen("__ESBMC_defined$" + id2string(F.first) + "$");

  for (auto it = body.instructions.begin(); it != body.instructions.end(); ++it)
  {
    if (it->is_decl())
    {
      const code_decl2t &decl = to_code_decl2t(it->code);
      const symbolt *s = context.find_symbol(decl.value);
      type2tc shadow_t = s ? tracked_shadow_type(*s) : type2tc();
      if (!is_nil_type(shadow_t) && !shadows.count(decl.value))
      {
        symbolt &shadow =
          gen.new_symbol(context, migrate_type_back(shadow_t), "");
        shadows.emplace(decl.value, shadow.id);
        by_sid.emplace(shadow.id, shadow_record{shadow_t, id2string(s->name)});

        // Insert immediately after the DECL:
        //   DECL shadow;
        //   ASSIGN shadow = false   (or constant_array_of(false) for arrays)
        auto pos = std::next(it);
        auto decl_it = body.instructions.insert(pos, DECL);
        decl_it->code = code_decl2tc(shadow_t, shadow.id);
        decl_it->location = it->location;
        decl_it->function = it->function;

        auto assign_it = body.instructions.insert(pos, ASSIGN);
        assign_it->code = code_assign2tc(
          symbol2tc(shadow_t, shadow.id),
          shadow_const(shadow_t, gen_false_expr()));
        assign_it->location = it->location;
        assign_it->function = it->function;

        ++it; // skip past inserted DECL shadow
        ++it; // skip past inserted ASSIGN shadow = init
        continue;
      }
    }

    collected co;
    write_target wt;

    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      wt = direct_write_target(a.target, shadows, by_sid);
      if (wt.root.empty())
        collect(a.target, false, shadows, by_sid, co);
      else if (!is_nil_expr(wt.index))
        collect(wt.index, false, shadows, by_sid, co);
      collect(a.source, false, shadows, by_sid, co);
    }
    else if (it->is_function_call())
    {
      const code_function_call2t &fc = to_code_function_call2t(it->code);
      wt = direct_write_target(fc.ret, shadows, by_sid);
      if (wt.root.empty())
        collect(fc.ret, false, shadows, by_sid, co);
      else if (!is_nil_expr(wt.index))
        collect(wt.index, false, shadows, by_sid, co);
      collect(fc.function, false, shadows, by_sid, co);
      for (const auto &arg : fc.operands)
        collect(arg, false, shadows, by_sid, co);
    }
    else
    {
      collect(it->guard, false, shadows, by_sid, co);
      collect(it->code, false, shadows, by_sid, co);
    }

    // Build a prefix-program of instructions to splice in *before* `it`.
    // insert_swap preserves jumps that target `it` by moving the existing
    // content to the next position.
    goto_programt new_code;

    for (const irep_idt &sid : co.whole_flips)
    {
      const shadow_record &rec = by_sid.at(sid);
      auto t = new_code.add_instruction(ASSIGN);
      t->code = code_assign2tc(
        symbol2tc(rec.type, sid), shadow_const(rec.type, gen_true_expr()));
      t->location = it->location;
      t->function = it->function;
    }

    auto emit_assert = [&](const expr2tc &guard, const shadow_record &rec) {
      auto t = new_code.add_instruction(ASSERT);
      t->guard = guard;
      t->location = it->location;
      t->location.comment("use of uninitialized variable: " + rec.user_name);
      t->location.property("uninitialised-variable");
      t->function = it->function;
    };

    for (const irep_idt &sid : co.scalar_reads)
      emit_assert(symbol2tc(get_bool_type(), sid), by_sid.at(sid));

    for (const auto &[sid, idx] : co.indexed_reads)
    {
      const shadow_record &rec = by_sid.at(sid);
      emit_assert(
        index2tc(get_bool_type(), symbol2tc(rec.type, sid), idx), rec);
    }

    while (!new_code.instructions.empty())
    {
      body.insert_swap(it, new_code.instructions.front());
      new_code.instructions.pop_front();
      ++it;
    }

    // After a direct write to a tracked local, flip the matching shadow
    // bit(s) to true. Insert *after* `it`; jumps and labels on `it` are
    // unaffected.
    if (wt.root.empty())
      continue;
    irep_idt sid = shadow_of(wt.root, shadows);
    if (sid.empty())
      continue;
    const shadow_record &rec = by_sid.at(sid);
    expr2tc shadow_sym = symbol2tc(rec.type, sid);
    expr2tc lhs = shadow_sym;
    expr2tc rhs = gen_true_expr();
    if (is_array_type(rec.type))
    {
      if (is_nil_expr(wt.index))
        rhs = constant_array_of2tc(rec.type, gen_true_expr());
      else
        lhs = index2tc(get_bool_type(), shadow_sym, wt.index);
    }
    auto pos = std::next(it);
    auto t = body.instructions.insert(pos, ASSIGN);
    t->code = code_assign2tc(lhs, rhs);
    t->location = it->location;
    t->function = it->function;
    ++it;
  }

  return !shadows.empty();
}
