#include <goto-programs/python_exception_reachability.h>
#include <goto-programs/goto_functions.h>
#include <goto-programs/remove_exceptions.h>
#include <irep2/irep2_utils.h>

#include <functional>
#include <set>

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
  for (const auto &[id, fn] : goto_functions.function_map)
  {
    if (!fn.body_available || !reachable.count(id))
      continue;
    for (const goto_programt::instructiont &i : fn.body.instructions)
      if (
        (i.is_assert() && !is_true(i.guard) && !is_internal_check(i)) ||
        (!allow_loops && i.is_backwards_goto()))
        return false;
  }
  return true;
}
