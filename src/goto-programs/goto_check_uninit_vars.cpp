#include <goto-programs/goto_check_uninit_vars.h>

#include <map>
#include <set>
#include <util/c_types.h>
#include <util/migrate.h>
#include <util/prefix.h>
#include <util/symbol_generator.h>

namespace
{
/// Tracked iff scalar, automatic storage, lvalue, not internal/return.
bool is_tracked_local(const symbolt &s)
{
  if (s.static_lifetime || s.is_extern || !s.lvalue)
    return false;
  if (has_prefix(s.name, "return_value$"))
    return false;
  if (has_prefix(s.id, "__ESBMC_") || has_prefix(s.name, "__ESBMC_"))
    return false;
  // Scalar types only — pointers / aggregates are out of scope for CWE-457
  // (pointer uninitialised state is reported under CWE-908 already).
  return s.type.is_bool() || s.type.id() == "signedbv" ||
         s.type.id() == "unsignedbv" || s.type.id() == "fixedbv" ||
         s.type.id() == "floatbv";
}

/// LHS name of a direct-symbol ASSIGN, or empty for compound LHS, nil,
/// or any non-symbol expression.
irep_idt direct_symbol(const expr2tc &lhs)
{
  if (is_nil_expr(lhs))
    return irep_idt();
  if (is_symbol2t(lhs))
    return to_symbol2t(lhs).thename;
  return irep_idt();
}

/// Lookup helper: return the shadow id for `name` if tracked, else empty.
irep_idt
shadow_of(irep_idt name, const std::map<irep_idt, irep_idt> &shadows)
{
  auto it = shadows.find(name);
  return it == shadows.end() ? irep_idt() : it->second;
}

/// Walk an expression, collecting:
/// - every direct read of a tracked symbol (visited as `reads`),
/// - every `&local` where the address is taken (visited as `address_takes`).
///
/// `lvalue_ctx` is true while recursing inside an l-value context (the
/// operand of an `address_of2t`, or the source of a `member2t` /
/// `index2t.source_value` chain whose root is an address-of). In that mode
/// the base symbol is not a value-read; only embedded sub-expressions —
/// for example `i` in `a[i]` — are reads.
void collect(
  const expr2tc &expr,
  bool lvalue_ctx,
  const std::map<irep_idt, irep_idt> &shadows,
  std::set<irep_idt> &reads,
  std::set<irep_idt> &address_takes)
{
  if (is_nil_expr(expr))
    return;

  if (is_address_of2t(expr))
  {
    const expr2tc &target = to_address_of2t(expr).ptr_obj;
    // Record the address-take for the root symbol of the l-value chain.
    irep_idt root;
    const expr2tc *cur = &target;
    while (true)
    {
      if (is_symbol2t(*cur))
      {
        root = to_symbol2t(*cur).thename;
        break;
      }
      if (is_member2t(*cur))
        cur = &to_member2t(*cur).source_value;
      else if (is_index2t(*cur))
        cur = &to_index2t(*cur).source_value;
      else
        break;
    }
    if (!root.empty())
    {
      irep_idt sid = shadow_of(root, shadows);
      if (!sid.empty())
        address_takes.insert(sid);
    }
    // Recurse into target in l-value mode so embedded reads (e.g. the
    // index of `&a[i]`) are still checked.
    collect(target, true, shadows, reads, address_takes);
    return;
  }

  if (is_symbol2t(expr))
  {
    if (lvalue_ctx)
      return; // base symbol of an l-value chain; not a value-read
    irep_idt sid = shadow_of(to_symbol2t(expr).thename, shadows);
    if (!sid.empty())
      reads.insert(sid);
    return;
  }

  if (lvalue_ctx && is_index2t(expr))
  {
    const index2t &i = to_index2t(expr);
    collect(i.source_value, true, shadows, reads, address_takes);
    collect(i.index, false, shadows, reads, address_takes); // index IS a read
    return;
  }

  if (lvalue_ctx && is_member2t(expr))
  {
    collect(
      to_member2t(expr).source_value, true, shadows, reads, address_takes);
    return; // member name is a literal
  }

  // In l-value mode, a `dereference2t` exits the l-value context: the
  // pointer operand is itself a value-read.
  if (lvalue_ctx && is_dereference2t(expr))
  {
    collect(
      to_dereference2t(expr).value, false, shadows, reads, address_takes);
    return;
  }

  // Default: preserve the current context for unknown wrappers (e.g. an
  // if-then-else lvalue `&(c ? a : b)` keeps `a` and `b` in l-value mode).
  // In r-value contexts this still descends r-value, which is what we want
  // for ordinary expressions like `a + b`. Preserving the context avoids
  // treating a base lvalue symbol as a value-read — a false-positive risk
  // we cannot tolerate as a formal verifier.
  expr->foreach_operand(
    [&](const expr2tc &e)
    { collect(e, lvalue_ctx, shadows, reads, address_takes); });
}
} // namespace

bool goto_check_uninit_vars::runOnFunction(
  std::pair<const dstring, goto_functiont> &F)
{
  if (!F.second.body_available)
    return false;

  goto_programt &body = F.second.body;
  std::map<irep_idt, irep_idt> shadows;           // user-name -> shadow_id
  std::map<irep_idt, std::string> shadow_display; // shadow_id -> display
  // Prefix uppercase so `is_tracked_local`'s "__ESBMC_" filter at the next
  // pass invocation rejects our own shadows — defence in depth against
  // self-instrumentation on a re-run.
  symbol_generator gen("__ESBMC_defined$" + id2string(F.first) + "$");

  for (auto it = body.instructions.begin(); it != body.instructions.end(); ++it)
  {
    if (it->is_decl())
    {
      const code_decl2t &decl = to_code_decl2t(it->code);
      const symbolt *s = context.find_symbol(decl.value);
      if (s && is_tracked_local(*s) && !shadows.count(decl.value))
      {
        symbolt &shadow =
          gen.new_symbol(context, migrate_type_back(get_bool_type()), "");
        shadows.emplace(decl.value, shadow.id);
        shadow_display.emplace(shadow.id, id2string(s->name));

        // Insert immediately after the DECL: DECL shadow; ASSIGN shadow = false;
        auto pos = std::next(it);
        auto decl_it = body.instructions.insert(pos, DECL);
        decl_it->code = code_decl2tc(get_bool_type(), shadow.id);
        decl_it->location = it->location;
        decl_it->function = it->function;

        auto assign_it = body.instructions.insert(pos, ASSIGN);
        assign_it->code = code_assign2tc(
          symbol2tc(get_bool_type(), shadow.id), gen_false_expr());
        assign_it->location = it->location;
        assign_it->function = it->function;

        ++it; // skip past inserted DECL shadow
        ++it; // skip past inserted ASSIGN shadow = false
        continue;
      }
    }

    // Collect reads and address-takes from every expression on the
    // instruction. Direct assignments and function-call returns treat the
    // LHS bare-symbol as a *write*, not a read.
    std::set<irep_idt> reads;
    std::set<irep_idt> address_takes;
    irep_idt write_target;

    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      write_target = direct_symbol(a.target);
      if (write_target.empty())
        collect(a.target, false, shadows, reads, address_takes);
      collect(a.source, false, shadows, reads, address_takes);
    }
    else if (it->is_function_call())
    {
      const code_function_call2t &fc = to_code_function_call2t(it->code);
      write_target = direct_symbol(fc.ret);
      if (write_target.empty())
        collect(fc.ret, false, shadows, reads, address_takes);
      collect(fc.function, false, shadows, reads, address_takes);
      for (const auto &arg : fc.operands)
        collect(arg, false, shadows, reads, address_takes);
    }
    else
    {
      collect(it->guard, false, shadows, reads, address_takes);
      collect(it->code, false, shadows, reads, address_takes);
    }

    // Build a prefix-program of instructions to splice in *before* `it`.
    // insert_swap preserves jumps that target `it` by moving the existing
    // content to the next position.
    goto_programt new_code;

    for (const irep_idt &shadow_id : address_takes)
    {
      goto_programt::targett t = new_code.add_instruction(ASSIGN);
      t->code =
        code_assign2tc(symbol2tc(get_bool_type(), shadow_id), gen_true_expr());
      t->location = it->location;
      t->function = it->function;
    }

    for (const irep_idt &shadow_id : reads)
    {
      auto dit = shadow_display.find(shadow_id);
      const std::string name =
        dit != shadow_display.end() ? dit->second : id2string(shadow_id);
      goto_programt::targett t = new_code.add_instruction(ASSERT);
      t->guard = symbol2tc(get_bool_type(), shadow_id);
      t->location = it->location;
      t->location.comment("use of uninitialized variable: " + name);
      t->location.property("uninitialised-variable");
      t->function = it->function;
    }

    while (!new_code.instructions.empty())
    {
      body.insert_swap(it, new_code.instructions.front());
      new_code.instructions.pop_front();
      ++it;
    }

    // After a direct write to a tracked local (ASSIGN or FUNCTION_CALL
    // return), flip its shadow to true. Insert *after* `it`; jumps and
    // labels on `it` are unaffected.
    if (!write_target.empty())
    {
      irep_idt sid = shadow_of(write_target, shadows);
      if (!sid.empty())
      {
        auto pos = std::next(it);
        auto t = body.instructions.insert(pos, ASSIGN);
        t->code =
          code_assign2tc(symbol2tc(get_bool_type(), sid), gen_true_expr());
        t->location = it->location;
        t->function = it->function;
        ++it;
      }
    }
  }

  return !shadows.empty();
}
