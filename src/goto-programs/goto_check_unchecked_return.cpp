#include <goto-programs/goto_check_unchecked_return.h>

#include <util/fallible_calls.h>
#include <util/prefix.h>

namespace
{
/// Return the bare symbol name if `e` is a direct-symbol expression,
/// possibly wrapped in a single typecast; empty otherwise.
irep_idt strip_cast_symbol(const expr2tc &e)
{
  if (is_nil_expr(e))
    return irep_idt();
  const expr2tc &inner = is_typecast2t(e) ? to_typecast2t(e).from : e;
  if (is_symbol2t(inner))
    return to_symbol2t(inner).thename;
  return irep_idt();
}

/// True iff `expr` reads the symbol named `tracked` anywhere in its
/// sub-expressions.
bool reads_symbol(const expr2tc &expr, const irep_idt &tracked)
{
  if (is_nil_expr(expr))
    return false;
  if (is_symbol2t(expr))
    return to_symbol2t(expr).thename == tracked;
  bool found = false;
  expr->foreach_operand(
    [&](const expr2tc &e)
    {
      if (!found && reads_symbol(e, tracked))
        found = true;
    });
  return found;
}

/// Bare C identifier of `id`, looked up in the symbol table. Returns
/// empty when the symbol resolves to a synthetic temporary (the Clang
/// frontend's `return_value$_<fn>$N` form, which is meaningless in a
/// user-facing diagnostic) so callers can drop the trailing `: <name>`.
std::string pretty_name(const contextt &ctx, const irep_idt &id)
{
  const symbolt *s = ctx.find_symbol(id);
  const std::string base =
    s && !s->name.empty() ? id2string(s->name) : id2string(id);
  if (has_prefix(base, "return_value$"))
    return std::string();
  return base;
}

void insert_assertion(
  goto_programt &body,
  goto_programt::targett use_it,
  const expr2tc &tracked,
  const fallible_call_t &fc,
  const std::string &fn_name,
  const std::string &tracked_display)
{
  goto_programt new_code;
  goto_programt::targett t = new_code.add_instruction(ASSERT);
  t->guard = success_predicate(fc.kind, tracked);
  t->location = use_it->location;
  std::string comment = "unchecked return value of " + fn_name;
  if (!tracked_display.empty())
    comment += ": " + tracked_display;
  t->location.comment(comment);
  t->location.property("unchecked-return-value");
  t->function = use_it->function;
  body.insert_swap(use_it, new_code.instructions.front());
}

/// Walks forward from the call instruction; emits at most one assertion
/// for this call, before the first use of the tracked return value that
/// is not protected by an upstream guard.
void scan_forward(
  contextt &context,
  goto_programt &body,
  goto_programt::targett call_it,
  const fallible_call_t &fc,
  const std::string &fn_name)
{
  const code_function_call2t &call = to_code_function_call2t(call_it->code);
  // v1 limitation: a call whose return value is discarded (no LHS) is not
  // instrumented — flagging it would require synthesising a fresh symbol
  // to bind the call's return. Tracked as a follow-up.
  if (is_nil_expr(call.ret) || !is_symbol2t(call.ret))
    return;
  irep_idt tracked = to_symbol2t(call.ret).thename;
  // The success predicate is always built against `call.ret` — the
  // synthetic temp the Clang frontend introduces — never against the
  // propagation target, whose type may differ (e.g. a downcast from
  // `ssize_t` to `int` would let `>= 0` vacuously hold). `tracked` is the
  // name we walk the CFG with; the predicate operand is fixed.
  const expr2tc predicate_value = call.ret;

  const auto emit = [&](goto_programt::targett at) {
    insert_assertion(
      body, at, predicate_value, fc, fn_name, pretty_name(context, tracked));
  };

  for (auto it = std::next(call_it); it != body.instructions.end(); ++it)
  {
    // Path-narrowing constructs that read the tracked value (GOTO guards,
    // __ESBMC_assume, an existing ASSERT) confine the assertion's path
    // condition, so we leave the use alone. An ASSERT the user already
    // wrote — or one a previous run of this pass inserted — is itself a
    // check, not an unchecked use.
    if (it->is_goto() || it->is_assume() || it->is_assert())
    {
      if (reads_symbol(it->guard, tracked))
        return;
      continue;
    }

    if (it->is_assign())
    {
      const code_assign2t &a = to_code_assign2t(it->code);

      // Pure propagation: `lhs = (cast)? tracked`. v1 tracks a single
      // alias hop — adequate for the `tmp = f(...); user = tmp` pattern
      // the Clang frontend emits.
      if (strip_cast_symbol(a.source) == tracked && is_symbol2t(a.target))
      {
        tracked = to_symbol2t(a.target).thename;
        continue;
      }

      if (reads_symbol(a.source, tracked) || reads_symbol(a.target, tracked))
      {
        emit(it);
        return;
      }

      if (is_symbol2t(a.target) && to_symbol2t(a.target).thename == tracked)
        return; // tracked variable overwritten with unrelated value
      continue;
    }

    if (it->is_function_call())
    {
      const code_function_call2t &fc2 = to_code_function_call2t(it->code);
      for (const expr2tc &arg : fc2.operands)
        if (reads_symbol(arg, tracked))
        {
          emit(it);
          return;
        }
      if (
        !is_nil_expr(fc2.ret) && is_symbol2t(fc2.ret) &&
        to_symbol2t(fc2.ret).thename == tracked)
        return; // tracked variable overwritten
      continue;
    }

    // RETURN, OTHER (e.g. FREE, dead-object marker) — any read of the
    // tracked symbol is an unchecked use.
    if (reads_symbol(it->code, tracked) || reads_symbol(it->guard, tracked))
    {
      emit(it);
      return;
    }

    if (it->is_end_function())
      return;
  }
}
} // namespace

bool goto_check_unchecked_return::runOnFunction(
  std::pair<const dstring, goto_functiont> &F)
{
  if (!F.second.body_available)
    return false;

  goto_programt &body = F.second.body;
  bool changed = false;

  // Single-pass iteration. `scan_forward` may use `insert_swap` to splice
  // an ASSERT before a later use; std::list iterators stay valid under
  // insertion, and any newly-inserted ASSERT we encounter on subsequent
  // iterations is skipped by the `is_function_call()` check.
  for (auto it = body.instructions.begin(); it != body.instructions.end(); ++it)
  {
    if (!it->is_function_call())
      continue;
    const code_function_call2t &call = to_code_function_call2t(it->code);
    if (!is_symbol2t(call.function))
      continue;
    const std::string fn_name =
      pretty_name(context, to_symbol2t(call.function).thename);
    const fallible_call_t *fc = find_fallible(fn_name);
    if (!fc)
      continue;
    // Report the canonical C name in the property comment (e.g.
    // `pthread_mutex_lock`) rather than the OM-mangled callee
    // (`pthread_mutex_lock_noassert`) the goto-program actually carries.
    const std::string canonical(fc->name);
    scan_forward(context, body, it, *fc, canonical);
    changed = true;
  }

  return changed;
}
