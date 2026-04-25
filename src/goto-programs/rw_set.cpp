#include <goto-programs/rw_set.h>

void rw_sett::compute(const expr2tc &expr)
{
  if (is_nil_expr(expr))
    return;

  if (is_code_assign2t(expr))
  {
    const code_assign2t &c = to_code_assign2t(expr);
    assign(c.target, c.source);
  }
  else if (is_code_printf2t(expr))
  {
    for (const auto &op : to_code_printf2t(expr).operands)
      read_rec(op);
  }
  else if (is_code_return2t(expr))
  {
    read_rec(to_code_return2t(expr).operand);
  }
  else if (is_code_function_call2t(expr))
  {
    const code_function_call2t &call = to_code_function_call2t(expr);

    // The return-value lhs (if any) is a write target.
    if (!is_nil_expr(call.ret))
      read_write_rec(
        call.ret, false, true, "", gen_true_expr(), expr2tc());

    // For function calls, we first check to see if the function has a body
    // available, and if so, we skip it because we also check inside the
    // function. If not, we need to check these args.
    if (is_symbol2t(call.function))
    {
      const symbolt *symbol = ns.lookup(to_symbol2t(call.function).thename);
      if (
        symbol &&
        (symbol->value.is_nil() || symbol->name == "__VERIFIER_assert"))
        for (const auto &arg : call.operands)
        {
          if (is_address_of2t(arg))
            read_rec(to_address_of2t(arg).ptr_obj);
          else
            read_rec(arg);
        }
    }
  }
  else
  {
    // Plain expression: only the goto/assert/assume guard is read.
    const goto_programt::instructiont &instruction = *target;
    if (
      instruction.is_goto() || instruction.is_assert() ||
      instruction.is_assume())
      read_rec(expr);
  }
}

void rw_sett::assign(const expr2tc &lhs, const expr2tc &rhs)
{
  read_rec(rhs);
  read_write_rec(lhs, false, true, "", gen_true_expr(), expr2tc());
}

void rw_sett::read_write_rec(
  const expr2tc &expr,
  bool r,
  bool w,
  const std::string &suffix,
  const expr2tc &guard,
  const expr2tc &original_expr,
  bool dereferenced)
{
  if (is_nil_expr(expr))
    return;

  if (is_symbol2t(expr))
  {
    const symbol2t &sym = to_symbol2t(expr);

    const symbolt *symbol = ns.lookup(sym.thename);
    if (symbol)
    {
      if (
        (!symbol->static_lifetime && !dereferenced) || symbol->is_thread_local)
        return; // ignore for now

      if (
        symbol->name == "__ESBMC_alloc" ||
        symbol->name == "__ESBMC_alloc_size" || symbol->name == "stdin" ||
        symbol->name == "stdout" || symbol->name == "stderr" ||
        symbol->name == "sys_nerr" || symbol->name == "operator=::ref" ||
        symbol->name == "this" || symbol->name == "__ESBMC_atexits")
        return; // ignore for now

      // Improvements for CUDA features
      if (symbol->name == "indexOfThread" || symbol->name == "indexOfBlock")
        return; // ignore for now

      // C11 _Atomic variables are never involved in data races
      // (§5.1.2.4p4: concurrent accesses to atomic objects are not races).
      if (symbol->type.get_bool("#atomic"))
        return;
    }

    irep_idt object = id2string(sym.thename) + suffix;

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
    const member2t &m = to_member2t(expr);
    expr2tc tmp = is_nil_expr(original_expr) ? expr : original_expr;
    read_write_rec(
      m.source_value,
      r,
      w,
      "." + id2string(m.member) + suffix,
      guard,
      tmp,
      dereferenced);
  }
  else if (is_index2t(expr))
  {
    const index2t &i = to_index2t(expr);
    expr2tc tmp = is_nil_expr(original_expr) ? expr : original_expr;
    read_write_rec(i.source_value, r, w, suffix, guard, tmp, dereferenced);
  }
  else if (is_dereference2t(expr))
  {
    const dereference2t &d = to_dereference2t(expr);
    expr2tc tmp = is_nil_expr(original_expr) ? expr : original_expr;
    read_write_rec(d.value, r, w, suffix, guard, tmp, true);
  }
  else if (is_address_of2t(expr))
  {
    // taking the address does not read or write the underlying object
  }
  else if (is_if2t(expr))
  {
    const if2t &ite = to_if2t(expr);
    read_rec(ite.cond, guard, original_expr);

    expr2tc true_guard = and2tc(guard, ite.cond);
    read_write_rec(
      ite.true_value, r, w, suffix, true_guard, original_expr, dereferenced);

    expr2tc false_guard = and2tc(guard, not2tc(ite.cond));
    read_write_rec(
      ite.false_value, r, w, suffix, false_guard, original_expr, dereferenced);
  }
  else
  {
    expr->foreach_operand([&](const expr2tc &op) {
      read_write_rec(op, r, w, suffix, guard, original_expr, dereferenced);
    });
  }
}
