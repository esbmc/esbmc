#include <goto-programs/python_exception_reachability.h>
#include <goto-programs/python_claim_discharge.h>
#include <goto-programs/goto_functions.h>
#include <goto-programs/remove_exceptions.h>
#include <irep2/irep2_utils.h>

#include <algorithm>
#include <functional>
#include <optional>
#include <set>
#include <vector>

namespace
{
// The float-printing model's own bounds checks, which no Python exception
// stands behind.
bool is_internal_check(const goto_programt::instructiont &i)
{
  return i.location.comment().as_string().find("__ESBMC_float_buf") !=
         std::string::npos;
}

// A call cycle needs an unwinding assertion in symex, which has no GOTO ASSERT.
// A call through a pointer may close one, so it counts as a cycle.
bool has_call_cycle(const goto_functionst &goto_functions)
{
  std::set<irep_idt> done, on_path;
  std::function<bool(const irep_idt &)> cycles_from = [&](const irep_idt &id) {
    if (on_path.count(id))
      return true;
    if (!done.insert(id).second)
      return false;
    auto fn = goto_functions.function_map.find(id);
    if (fn == goto_functions.function_map.end() || !fn->second.body_available)
      return false;
    on_path.insert(id);
    for (const auto &i : fn->second.body.instructions)
    {
      if (!i.is_function_call())
        continue;
      const expr2tc &callee = to_code_function_call2t(i.code).function;
      if (!is_symbol2t(callee) || cycles_from(to_symbol2t(callee).thename))
        return true;
    }
    on_path.erase(id);
    return false;
  };
  return cycles_from("__ESBMC_main");
}
// Symex lowers isinstance() and fails a claim of its own when the operand is a
// `void *` whose type it cannot trace (#10251). That claim has no GOTO
// ASSERT, so any isinstance() of a `void *` value keeps the normal run.
bool may_lose_isinstance_type(const expr2tc &e)
{
  if (is_nil_expr(e))
    return false;
  if (is_isinstance2t(e))
  {
    bool untyped = false;
    std::function<void(const expr2tc &)> scan = [&](const expr2tc &x) {
      if (is_nil_expr(x) || untyped)
        return;
      untyped = is_pointer_type(x->type) &&
                is_empty_type(to_pointer_type(x->type).subtype);
      x->foreach_operand(scan);
    };
    scan(to_isinstance2t(e).side_1);
    if (untyped)
      return true;
  }
  bool found = false;
  e->foreach_operand(
    [&](const expr2tc &o) { found = found || may_lose_isinstance_type(o); });
  return found;
}

/// The assertions of @p fn that can fail, or none when @p fn has something
/// that keeps the normal run.
std::optional<std::vector<goto_programt::const_targett>>
failable_claims(const goto_functiont &fn, bool allow_loops)
{
  std::vector<goto_programt::const_targett> claims;
  const auto &body = fn.body.instructions;
  for (auto i = body.begin(); i != body.end(); ++i)
  {
    if (
      (!allow_loops && i->is_backwards_goto()) ||
      may_lose_isinstance_type(i->code) || may_lose_isinstance_type(i->guard))
      return std::nullopt;
    if (i->is_assert() && !is_true(i->guard) && !is_internal_check(*i))
      claims.push_back(i);
  }
  return claims;
}
} // namespace

bool no_reachable_property(
  const goto_functionst &goto_functions,
  const contextt &context,
  bool allow_loops)
{
  const std::set<irep_idt> reachable =
    entry_reachable_functions(goto_functions, context);
  if (!allow_loops && has_call_cycle(goto_functions))
    return false;
  std::vector<goto_programt::const_targett> claims;
  for (const auto &[id, fn] : goto_functions.function_map)
  {
    if (!fn.body_available || !reachable.count(id))
      continue;
    const auto found = failable_claims(fn, allow_loops);
    if (!found)
      return false;
    claims.insert(claims.end(), found->begin(), found->end());
  }
  if (claims.empty())
    return true;
  const python_claim_dischargert discharge(goto_functions, reachable);
  return std::all_of(
    claims.begin(), claims.end(), [&](auto i) { return discharge.proves(i); });
}
