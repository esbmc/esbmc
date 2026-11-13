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

/// Whether an allocation of @p size may fail a claim: symex refuses a constant
/// size too large to place. A size that fits even a 16-bit target cannot.
bool is_unplaceable_size(expr2tc size)
{
  while (!is_nil_expr(size) && (is_typecast2t(size) || is_sizeof2t(size)))
    size =
      is_typecast2t(size) ? to_typecast2t(size).from : to_sizeof2t(size).value;
  if (is_nil_expr(size) || !is_constant_int2t(size))
    return false;
  const BigInt &v = to_constant_int2t(size).value;
  return v.is_negative() || v >= BigInt(1 << 15);
}

/// Whether evaluating @p e may make symex add a claim of its own: under
/// pointer checks a dereference may be null, invalid, freed, out of bounds or
/// misaligned, and so may a printf operand; whatever the options, va_arg
/// checks its list, realloc its operand, and an allocation its size.
bool expr_adds_symex_claim(const expr2tc &e, bool pointer_checks)
{
  if (is_nil_expr(e))
    return false;
  if (is_dereference2t(e) && pointer_checks)
    return true;
  if (is_sideeffect2t(e))
  {
    const sideeffect2t &s = to_sideeffect2t(e);
    using kind = sideeffect2t::allockind;
    if (
      s.kind == kind::va_arg || s.kind == kind::realloc ||
      (s.kind == kind::printf2 && pointer_checks))
      return true;
    if (
      (s.kind == kind::malloc || s.kind == kind::alloca ||
       s.kind == kind::cpp_new || s.kind == kind::cpp_new_arr) &&
      is_unplaceable_size(s.size))
      return true;
  }
  bool found = false;
  e->foreach_operand([&](const expr2tc &o) {
    found = found || expr_adds_symex_claim(o, pointer_checks);
  });
  return found;
}

/// Whether symex may add a claim of its own at @p i, with no GOTO ASSERT for
/// this pass to see. Besides the expressions above: free and delete check
/// their operand; under pointer checks, so do printf, a call through a
/// pointer (the pointer itself), and a call to a function with no body, whose
/// intrinsic may dereference its arguments.
bool adds_symex_claim(
  const goto_functionst &goto_functions,
  const goto_programt::instructiont &i,
  bool pointer_checks)
{
  if (
    may_lose_isinstance_type(i.code) || may_lose_isinstance_type(i.guard) ||
    expr_adds_symex_claim(i.code, pointer_checks) ||
    expr_adds_symex_claim(i.guard, pointer_checks))
    return true;
  if (is_nil_expr(i.code))
    return false;
  if (
    is_code_free2t(i.code) || is_code_cpp_delete2t(i.code) ||
    is_code_cpp_del_array2t(i.code))
    return true;
  if (!pointer_checks)
    return false;
  if (is_code_printf2t(i.code))
    return true;
  if (!i.is_function_call())
    return false;
  const expr2tc &callee = to_code_function_call2t(i.code).function;
  if (!is_symbol2t(callee))
    return true;
  auto fn = goto_functions.function_map.find(to_symbol2t(callee).thename);
  return fn == goto_functions.function_map.end() || !fn->second.body_available;
}

/// The assertions of @p fn that can fail, or none when @p fn has something
/// that keeps the normal run.
std::optional<std::vector<goto_programt::const_targett>> failable_claims(
  const goto_functionst &goto_functions,
  const goto_functiont &fn,
  bool allow_loops,
  bool pointer_checks)
{
  std::vector<goto_programt::const_targett> claims;
  const auto &body = fn.body.instructions;
  for (auto i = body.begin(); i != body.end(); ++i)
  {
    if (
      (!allow_loops && i->is_backwards_goto()) ||
      adds_symex_claim(goto_functions, *i, pointer_checks))
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
  bool allow_loops,
  bool pointer_checks)
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
    const auto found =
      failable_claims(goto_functions, fn, allow_loops, pointer_checks);
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
