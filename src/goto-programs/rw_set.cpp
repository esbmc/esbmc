#include <goto-programs/rw_set.h>
#include <pointer-analysis/goto_program_dereference.h>
#include <util/arith_tools.h>
#include <util/expr_util.h>
#include <util/namespace.h>
#include <util/std_expr.h>

// Follow pointer arithmetic and typecasts down to the root pointer/array/object
// symbol of an address expression (`A + i` -> A, `(T *)p` -> p), or return null
// when the base is not a plain symbol.
static const symbolt *root_object_symbol(const expr2tc &e, const namespacet &ns)
{
  expr2tc cur = e;
  while (!is_nil_expr(cur))
  {
    if (is_symbol2t(cur))
      return ns.lookup(to_symbol2t(cur).thename);
    if (is_typecast2t(cur))
      cur = to_typecast2t(cur).from;
    else if (is_add2t(cur))
      // pointer +/- integer offset: keep following the pointer-typed operand
      cur = is_pointer_type(to_add2t(cur).side_1->type) ? to_add2t(cur).side_1
                                                        : to_add2t(cur).side_2;
    else if (is_sub2t(cur))
      cur = to_sub2t(cur).side_1;
    else
      break;
  }
  return nullptr;
}

// C11 §5.1.2.4p4: concurrent accesses to _Atomic objects are not data races.
// irep2 has no atomic-type node (see the FIXME in clang_c_convert's Atomic
// case), so the `#atomic` flag survives only on the *legacy symbol type*. The
// per-symbol check in read_write_rec therefore misses an access that reaches an
// atomic element through a non-atomic pointer/array — e.g. `A[i]` with
// `_Atomic int *A`, where neither the pointer A nor the index i is itself
// atomic. Recover the flag from the base symbol's pointee/element type so such
// accesses are correctly excluded from race checking (issue #4431).
static bool accesses_atomic_object(const expr2tc &access, const namespacet &ns)
{
  if (is_nil_expr(access))
    return false;

  if (is_symbol2t(access))
  {
    const symbolt *s = ns.lookup(to_symbol2t(access).thename);
    return s && s->get_type().get_bool("#atomic");
  }

  if (is_dereference2t(access) || is_index2t(access))
  {
    const expr2tc &base = is_dereference2t(access)
                            ? to_dereference2t(access).value
                            : to_index2t(access).source_value;
    const symbolt *s = root_object_symbol(base, ns);
    if (!s)
      return false;
    const typet &t = s->get_type();
    if (t.is_pointer() || t.is_array())
      return t.subtype().get_bool("#atomic");
    return t.get_bool("#atomic");
  }

  if (is_member2t(access))
    return accesses_atomic_object(to_member2t(access).source_value, ns);

  return false;
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
    // An indirect call resolves its target by reading a function pointer (the
    // call target is a dereference such as `*fp`, not a function symbol). If
    // that pointer is shared, the read can race with a concurrent write, e.g.
    // `fp = f2` under a different lock (issue #4425). A direct call instead
    // names a function symbol (code, not data) and is handled below.
    if (is_dereference2t(code.function))
      // Read the pointer VALUE that selects the target, keyed on the pointer
      // object itself (&fp) so it aliases a concurrent write `fp = ...`.
      // Reading the dereference would key on the pointee `&(*fp)` and miss it.
      // For `(*tbl[i])()` this keys on the array object `tbl`.
      read_rec(to_dereference2t(code.function).value);
    else if (!is_symbol2t(code.function))
      read_rec(code.function);
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

    if (
      symbol->name == "__ESBMC_alloc" || symbol->name == "__ESBMC_alloc_size" ||
      symbol->name == "stdin" || symbol->name == "stdout" ||
      symbol->name == "stderr" || symbol->name == "sys_nerr" ||
      symbol->name == "operator=::ref" || symbol->name == "this" ||
      symbol->name == "__ESBMC_atexits")
    {
      return; // ignore for now
    }

    // Improvements for CUDA features
    if (symbol->name == "indexOfThread" || symbol->name == "indexOfBlock")
    {
      return; // ignore for now
    }

    // C11 _Atomic variables are never involved in data races
    // (§5.1.2.4p4: concurrent accesses to atomic objects are not races).
    if (symbol->get_type().get_bool("#atomic"))
      return;

    // The check above only sees this symbol's own type. When the access
    // reaches an atomic element through a non-atomic pointer/array (e.g.
    // `A[i]` with `_Atomic int *A`), the atomic object is named by
    // original_expr, not by this symbol; suppress those too (issue #4431).
    if (
      !is_nil_expr(original_expr) && accesses_atomic_object(original_expr, ns))
      return;

    irep_idt object = id2string(symbol_expr.thename) + suffix;

    entryt &entry = entries[object];
    entry.object = object;
    entry.r = entry.r || r;
    entry.w = entry.w || w;
    entry.deref = dereferenced;
    entry.guard = guard;
    entry.original_expr = is_nil_expr(original_expr) ? expr : original_expr;
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
    is_ieee_add2t(expr) || is_ieee_sub2t(expr) || is_ieee_mul2t(expr) ||
    is_ieee_div2t(expr) || is_ieee_fma2t(expr) || is_ieee_sqrt2t(expr))
  {
    // Same rationale: recurse into the FP value operands, never the rounding
    // mode. foreach_operand would also visit rounding_mode, so list the value
    // operands explicitly.
    std::vector<expr2tc> values;
    if (is_ieee_add2t(expr))
      values = {to_ieee_add2t(expr).side_1, to_ieee_add2t(expr).side_2};
    else if (is_ieee_sub2t(expr))
      values = {to_ieee_sub2t(expr).side_1, to_ieee_sub2t(expr).side_2};
    else if (is_ieee_mul2t(expr))
      values = {to_ieee_mul2t(expr).side_1, to_ieee_mul2t(expr).side_2};
    else if (is_ieee_div2t(expr))
      values = {to_ieee_div2t(expr).side_1, to_ieee_div2t(expr).side_2};
    else if (is_ieee_fma2t(expr))
      values = {
        to_ieee_fma2t(expr).value_1,
        to_ieee_fma2t(expr).value_2,
        to_ieee_fma2t(expr).value_3};
    else
      values = {to_ieee_sqrt2t(expr).value};

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
