#include <goto-programs/rw_set.h>
#include <pointer-analysis/goto_program_dereference.h>
#include <util/arith/arith_tools.h>
#include <util/expr/expr_util.h>
#include <util/symtab/namespace.h>
#include <util/irep/std_expr.h>
#include <unordered_set>

static bool is_untracked_global(const irep_idt &name)
{
  static const std::unordered_set<std::string> names = {
    "__ESBMC_alloc",
    "__ESBMC_alloc_size",
    "stdin",
    "stdout",
    "stderr",
    "sys_nerr",
    "operator=::ref",
    "this",
    "__ESBMC_atexit_count",
    "__ESBMC_atexit_registered",
    "__ESBMC_stdlib_atexit_key"};
  return names.count(name.as_string()) != 0;
}

// Follow pointer arithmetic and typecasts down to the pointer or array an
// address expression is based on (`A + i` -> A, `(T *)p` -> p).
static expr2tc pointer_base(expr2tc e)
{
  while (true)
  {
    if (is_typecast2t(e))
      e = to_typecast2t(e).from;
    else if (is_add2t(e))
      // pointer +/- integer offset: keep following the pointer-typed operand
      e = is_pointer_type(to_add2t(e).side_1->type) ? to_add2t(e).side_1
                                                    : to_add2t(e).side_2;
    else if (is_sub2t(e))
      e = to_sub2t(e).side_1;
    else
      return e;
  }
}

// C11 §5.1.2.4p4: concurrent accesses to _Atomic objects are not data races.
// irep2 has no atomic-type node (see the FIXME in clang_c_convert's Atomic
// case), so the `#atomic` flag survives only on legacy types. Rebuild the
// legacy type of the lvalue `e` from the symbol table, setting `atomic` when it
// or an object containing it is _Atomic. An _Atomic pointer does not make its
// pointee atomic. Returns a nil type when the type cannot be recovered.
static typet
lvalue_legacy_type(const expr2tc &e, const namespacet &ns, bool &atomic)
{
  typet t;
  if (is_symbol2t(e))
  {
    const symbolt *s = ns.lookup(to_symbol2t(e).thename);
    if (s)
      t = s->get_type();
  }
  else if (is_member2t(e))
  {
    const typet src =
      ns.follow(lvalue_legacy_type(to_member2t(e).source_value, ns, atomic));
    if (src.is_struct() || src.is_union())
      t = to_struct_union_type(src).get_component(to_member2t(e).member).type();
  }
  else if (is_index2t(e))
  {
    const typet src =
      ns.follow(lvalue_legacy_type(to_index2t(e).source_value, ns, atomic));
    if (src.is_array())
      t = src.subtype();
  }
  else if (is_dereference2t(e))
  {
    bool pointer_atomic = false;
    const typet src = ns.follow(lvalue_legacy_type(
      pointer_base(to_dereference2t(e).value), ns, pointer_atomic));
    if (src.is_array() || src.is_pointer())
      t = src.subtype();
  }
  atomic = atomic || t.get_bool("#atomic");
  return t;
}

static bool accesses_atomic_object(const expr2tc &access, const namespacet &ns)
{
  bool atomic = false;
  lvalue_legacy_type(access, ns, atomic);
  return atomic;
}

void rw_sett::compute(const expr2tc &expr)
{
  if (is_nil_expr(expr))
    return;

  const goto_programt::instructiont &instruction = *target;

  if (is_code_assign2t(expr))
  {
    const code_assign2t &code = to_code_assign2t(expr);
    assign(code.target, code.source);
  }
  else if (is_code_printf2t(expr))
  {
    for (const expr2tc &op : to_code_printf2t(expr).operands)
      read_rec(op);
  }
  else if (is_code_return2t(expr))
  {
    read_rec(to_code_return2t(expr).operand);
  }
  else if (is_code_function_call2t(expr))
  {
    const code_function_call2t &code = to_code_function_call2t(expr);
    read_write_rec(code.ret, false, true, "", guard2tc(), expr2tc());
    // An indirect call through a global function pointer reads that pointer to
    // resolve its target (the call target is `*fp`, not a function symbol). If
    // the pointer is shared, the read can race with a concurrent write, e.g.
    // `fp = f2` under a different lock (issue #4425). Record that read, keyed
    // on the pointer object itself (&fp) so it aliases the write `fp = ...` --
    // reading the dereference would key on the pointee `&(*fp)` and miss it.
    //
    // Restrict this to a plain pointer *variable* (`*fp`, or a cast of one such
    // as `*(fn_t)fp`, where fp is a symbol). Function pointers reached through
    // an aggregate (`*s.fp`, `*tbl[i]`) are in practice internal dispatch
    // tables -- e.g. the CUDA kernel-launch structs `dev_*.func` in
    // call_kernel.h, or the pthread cleanup stack -- written before the threads
    // that read them are even spawned, so they never race. Recording those
    // reads only inserts spurious yield() interleaving points that dilute
    // context-bounded search and hide real races; this mirrors the restriction
    // collect_thread_escaped_locals applies to address-taken locals for the
    // same reason. A direct call names a function symbol (code, not data) and
    // is left to the body-less-call branch below.
    if (is_dereference2t(code.function))
    {
      // Peel casts so a cast-wrapped pointer variable is still recognised, but
      // stop at the first aggregate/computed node so dispatch tables stay out.
      expr2tc target = to_dereference2t(code.function).value;
      while (is_typecast2t(target))
        target = to_typecast2t(target).from;
      if (is_symbol2t(target))
        read_rec(target);
    }
    // For function calls, we first check to see if the function has a body
    // available, and if so, we skip it because we also check inside the
    // function. If not, we need to check these args.
    if (is_symbol2t(code.function))
    {
      const symbolt *symbol = ns.lookup(to_symbol2t(code.function).thename);
      if (symbol->get_value().is_nil() || symbol->name == "__VERIFIER_assert")
        for (const expr2tc &arg : code.operands)
        {
          if (is_address_of2t(arg))
            read_rec(to_address_of2t(arg).ptr_obj);
          else
            read_rec(arg);
        }
    }
  }
  else if (
    instruction.is_goto() || instruction.is_assert() || instruction.is_assume())
    read_rec(expr);
}

void rw_sett::assign(const expr2tc &lhs, const expr2tc &rhs)
{
  read_rec(rhs);
  read_write_rec(lhs, false, true, "", guard2tc(), expr2tc());
}

// The FP nodes carry __ESBMC_rounding_mode as an extra operand that the legacy
// exprt form did not expose positionally, so a foreach_operand walk would
// register the rounding mode as a shared object. Return the value operands
// alone, or nothing when expr is not one of those nodes.
static std::vector<expr2tc> ieee_value_operands(const expr2tc &expr)
{
  if (is_ieee_add2t(expr))
    return {to_ieee_add2t(expr).side_1, to_ieee_add2t(expr).side_2};
  if (is_ieee_sub2t(expr))
    return {to_ieee_sub2t(expr).side_1, to_ieee_sub2t(expr).side_2};
  if (is_ieee_mul2t(expr))
    return {to_ieee_mul2t(expr).side_1, to_ieee_mul2t(expr).side_2};
  if (is_ieee_div2t(expr))
    return {to_ieee_div2t(expr).side_1, to_ieee_div2t(expr).side_2};
  if (is_ieee_rem2t(expr))
    return {to_ieee_rem2t(expr).side_1, to_ieee_rem2t(expr).side_2};
  if (is_ieee_fma2t(expr))
    return {
      to_ieee_fma2t(expr).value_1,
      to_ieee_fma2t(expr).value_2,
      to_ieee_fma2t(expr).value_3};
  if (is_ieee_sqrt2t(expr))
    return {to_ieee_sqrt2t(expr).value};
  return {};
}

void rw_sett::read_write_rec(
  const expr2tc &expr,
  bool r,
  bool w,
  const std::string &suffix,
  const guard2tc &guard,
  const expr2tc &original_expr,
  bool dereferenced)
{
  if (is_nil_expr(expr))
    return;

  if (is_symbol2t(expr))
  {
    const symbol2t &symbol_expr = to_symbol2t(expr);

    const symbolt *symbol = ns.lookup(symbol_expr.thename);
    // irep2 represents synthetic pointer constants (notably the NULL pointer)
    // as a symbol2t with no namespace entry; the legacy exprt form was a
    // constant_exprt, which never reached this symbol path. Skip them so we do
    // not register a bogus race object (and later take its address).
    if (!symbol)
      return;

    // Python module-level globals carry static_lifetime=false to keep
    // them out of the C-side static-init pass (their values double as
    // const-prop snapshots in the Python frontend). The Python frontend
    // sets file_local=false on them so this filter recognises them
    // as race-eligible shared state.
    const bool python_global = symbol->mode == "Python" && !symbol->file_local;
    // A non-static local whose address is taken may alias a pointer passed to
    // another thread (e.g. via pthread_create), so a direct access to it by
    // name is still race-relevant even though it is not a global (issue #4424).
    const bool address_escaped =
      shared_locals && shared_locals->count(symbol_expr.thename);
    if (
      (!symbol->static_lifetime && !dereferenced && !python_global &&
       !address_escaped) ||
      symbol->is_thread_local)
    {
      return; // ignore for now
    }

    if (is_untracked_global(symbol->name))
      return; // ignore for now

    // Improvements for CUDA features
    if (symbol->name == "indexOfThread" || symbol->name == "indexOfBlock")
    {
      return; // ignore for now
    }

    // The accessed object is original_expr when this symbol only names its
    // base, e.g. `A[i]` with `_Atomic int *A` (#4431) or `s.f`. An _Atomic
    // variable is never accessed non-atomically, so it needs no race flag. An
    // atomic member or element shares bytes with plain accesses to its
    // container, so it keeps its flag and only skips its own assertion.
    const expr2tc &access = is_nil_expr(original_expr) ? expr : original_expr;
    const bool atomic = accesses_atomic_object(access, ns);
    if (atomic && is_nil_expr(original_expr))
      return;

    irep_idt object = id2string(symbol_expr.thename) + suffix;

    entryt &entry = entries[object];
    entry.object = object;
    entry.r = entry.r || r;
    entry.w = entry.w || w;
    entry.deref = dereferenced;
    entry.atomic = atomic;
    entry.guard = guard;
    entry.original_expr = access;
  }
  else if (is_member2t(expr))
  {
    const member2t &member = to_member2t(expr);
    expr2tc tmp = is_nil_expr(original_expr) ? expr : original_expr;

    read_write_rec(
      member.source_value,
      r,
      w,
      "." + id2string(member.member) + suffix,
      guard,
      tmp,
      dereferenced);
  }
  else if (is_index2t(expr))
  {
    expr2tc tmp = is_nil_expr(original_expr) ? expr : original_expr;
    read_write_rec(
      to_index2t(expr).source_value, r, w, suffix, guard, tmp, dereferenced);
  }
  else if (is_dereference2t(expr))
  {
    expr2tc tmp = is_nil_expr(original_expr) ? expr : original_expr;
    read_write_rec(
      to_dereference2t(expr).value, r, w, suffix, guard, tmp, true);
  }
  else if (is_address_of2t(expr))
  {
    // Taking an address neither reads nor writes the pointee.
  }
  else if (is_if2t(expr))
  {
    const if2t &if_expr = to_if2t(expr);
    read_rec(if_expr.cond, guard, original_expr);

    guard2tc true_guard(guard);
    true_guard.add(if_expr.cond);
    read_write_rec(
      if_expr.true_value,
      r,
      w,
      suffix,
      true_guard,
      original_expr,
      dereferenced);

    guard2tc false_guard(guard);
    false_guard.add(not2tc(if_expr.cond));
    read_write_rec(
      if_expr.false_value,
      r,
      w,
      suffix,
      false_guard,
      original_expr,
      dereferenced);
  }
  else if (is_typecast2t(expr) || is_nearbyint2t(expr))
  {
    // Floating-point typecast/nearbyint carry __ESBMC_rounding_mode as an
    // extra operand that the legacy exprt form does not expose positionally.
    // Recurse only into the value operand so the operand walk matches the
    // pre-migration behaviour (and does not register the rounding mode as a
    // shared object). A genuine read/write of the rounding-mode global still
    // reaches the symbol path via assign()/read_rec().
    const expr2tc &from = is_typecast2t(expr) ? to_typecast2t(expr).from
                                              : to_nearbyint2t(expr).from;
    read_write_rec(from, r, w, suffix, guard, original_expr, dereferenced);
  }
  else if (
    std::vector<expr2tc> values = ieee_value_operands(expr); !values.empty())
  {
    for (const expr2tc &v : values)
      read_write_rec(v, r, w, suffix, guard, original_expr, dereferenced);
  }
  else
  {
    expr->foreach_operand([&](const expr2tc &op) {
      read_write_rec(op, r, w, suffix, guard, original_expr, dereferenced);
    });
  }
}
