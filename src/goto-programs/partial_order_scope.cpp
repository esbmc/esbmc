#include <goto-programs/partial_order_scope.h>
#include <irep2/irep2_utils.h>
#include <util/base/filesystem.h>
#include <util/message/message.h>

#include <set>
#include <string>

namespace
{
const std::string fn_prefix = "c:@F@";

std::string callee_name(const code_function_call2t &call)
{
  if (!is_symbol2t(call.function))
    return "";
  const std::string name = to_symbol2t(call.function).thename.as_string();
  return name.starts_with(fn_prefix) ? name.substr(fn_prefix.size()) : name;
}

bool is_supported_thread_call(const std::string &name)
{
  static const std::set<std::string> supported{
    "pthread_create",
    "pthread_join",
    "pthread_mutex_init",
    "pthread_mutex_lock",
    "pthread_mutex_unlock",
    // What the C frontend renames them to without --deadlock-check.
    "pthread_join_noswitch",
    "pthread_mutex_lock_noassert",
    "pthread_mutex_unlock_noassert",
    "pthread_mutex_destroy",
    "__ESBMC_atomic_begin",
    "__ESBMC_atomic_end",
    "__VERIFIER_atomic_begin",
    "__VERIFIER_atomic_end",
    // Inserted ahead of every atomic section; each thread runs to completion,
    // so it does nothing.
    "__ESBMC_yield"};
  return supported.count(name) != 0;
}

/// Calls whose meaning depends on the scheduler or on a thread API the
/// prototype does not model yet.
bool is_unsupported_call(const std::string &name)
{
  if (is_supported_thread_call(name))
    return false;
  for (const char *prefix : {"pthread_", "sem_", "thrd_", "mtx_", "cnd_"})
    if (name.starts_with(prefix))
      return true;
  static const std::set<std::string> scheduler_intrinsics{
    "__ESBMC_switch_to",
    "__ESBMC_switch_away_from",
    "__ESBMC_spawn_thread",
    "__ESBMC_terminate_thread",
    "__ESBMC_really_atomic_begin",
    "__ESBMC_really_atomic_end",
    "__ESBMC_switch_to_monitor",
    "__ESBMC_switch_from_monitor",
    "__ESBMC_register_monitor",
    "__ESBMC_kill_monitor"};
  return scheduler_intrinsics.count(name) != 0;
}

bool is_null(expr2tc e)
{
  while (is_typecast2t(e))
    e = to_typecast2t(e).from;
  return is_constant(e) && !is_constant_string2t(e);
}

/// Locks, which the memory model does not order yet, and assumptions, which
/// the equation applies to every later thread's claims whatever the clocks
/// say.
bool needs_unbuilt_events(const std::string &name)
{
  return name.starts_with("pthread_mutex_lock") || name == "__VERIFIER_assume";
}

class scope_checkt
{
public:
  scope_checkt(const namespacet &ns, bool verifying)
    : ns(ns), verifying(verifying)
  {
  }

  bool check(const goto_functionst &goto_functions)
  {
    functions = &goto_functions;
    collect_spawners(goto_functions);
    for (const auto &[name, fn] : goto_functions.function_map)
    {
      if (!is_user_function(name.as_string(), fn))
        continue;
      for (auto it = fn.body.instructions.begin();
           it != fn.body.instructions.end();
           ++it)
      {
        where = &it->location;
        if (!check_instruction(*it) || !check_loop(fn.body, it))
          return false;
      }
    }
    return true;
  }

private:
  const namespacet &ns;
  const bool verifying;
  const locationt *where = nullptr;
  const goto_functionst *functions = nullptr;
  /// Functions that create a thread, directly or through a call.
  std::set<std::string> spawners;

  static bool
  is_user_function(const std::string &name, const goto_functiont &fn)
  {
    return fn.body_available && !fn.body.hide &&
           name.find("__ESBMC") == std::string::npos &&
           !fn.body.instructions.empty() &&
           !file_operations::is_bundled_source(
             fn.body.instructions.front().location.get_file().as_string());
  }

  bool reject(const std::string &what) const
  {
    log_error(
      "--partial-order does not support {} yet, at {}",
      what,
      where->as_string());
    return false;
  }

  void collect_spawners(const goto_functionst &goto_functions)
  {
    spawners.insert(fn_prefix + "pthread_create");
    for (bool grew = true; grew;)
    {
      grew = false;
      for (const auto &[name, fn] : goto_functions.function_map)
        if (!spawners.count(name.as_string()) && calls_spawner(fn.body))
          grew = spawners.insert(name.as_string()).second;
    }
  }

  bool calls_spawner(const goto_programt &body) const
  {
    for (const auto &instruction : body.instructions)
      if (is_spawning_call(instruction))
        return true;
    return false;
  }

  bool is_spawning_call(const goto_programt::instructiont &instruction) const
  {
    if (!instruction.is_function_call())
      return false;
    const expr2tc &function =
      to_code_function_call2t(instruction.code).function;
    return is_symbol2t(function) &&
           spawners.count(to_symbol2t(function).thename.as_string());
  }

  /// One thread per unwinding of a loop belongs to phase 1.
  bool check_loop(const goto_programt &body, goto_programt::const_targett latch)
    const
  {
    if (!latch->is_backwards_goto())
      return true;
    for (auto it = latch->targets.front(); it != latch; ++it)
      if (is_spawning_call(*it))
        return reject("a thread created inside a loop");
    return true;
  }

  bool check_instruction(const goto_programt::instructiont &instruction) const
  {
    if (instruction.is_function_call())
      return check_call(to_code_function_call2t(instruction.code));
    if (verifying && instruction.is_assume())
      return reject("verifying an assumption");
    return check_expr(instruction.code) && check_expr(instruction.guard);
  }

  bool check_call(const code_function_call2t &call) const
  {
    const std::string name = callee_name(call);
    if (!is_symbol2t(call.function))
      return reject("a call through a function pointer");
    if (is_unsupported_call(name))
      return reject("a call to " + name);
    if (verifying && needs_unbuilt_events(name))
      return reject("verifying a call to " + name);

    if (!check_expr(call.ret) || !check_thread_args(name, call))
      return false;

    for (const expr2tc &arg : call.operands)
    {
      // The thread id and the mutex of a pthread call are passed by address.
      if (
        name.starts_with("pthread_") && is_address_of2t(arg) &&
        is_symbol2t(to_address_of2t(arg).ptr_obj))
        continue;
      // A symex intrinsic writes through the pointer without recording events.
      if (is_shared_address(arg) && !has_body(call))
        return reject("passing the address of a global to " + name);
      if (!check_expr(arg))
        return false;
    }
    return true;
  }

  bool has_body(const code_function_call2t &call) const
  {
    auto it = functions->function_map.find(to_symbol2t(call.function).thename);
    return it != functions->function_map.end() && it->second.body_available;
  }

  bool check_thread_args(
    const std::string &name,
    const code_function_call2t &call) const
  {
    const bool create = name == "pthread_create";
    const bool join = name.starts_with("pthread_join");
    if (create && !is_null(call.operands.at(3)))
      return reject("a thread argument other than NULL");
    if (join && !is_null(call.operands.at(1)))
      return reject("a join result pointer other than NULL");
    // Joins are ordered by a thread id known during symex, and a global's
    // value is chosen by the solver.
    if (
      (create && is_shared_address(call.operands.at(0))) ||
      (join && reads_shared(call.operands.at(0))))
      return reject("a thread id stored in a global");
    return true;
  }

  bool check_expr(const expr2tc &e) const
  {
    if (is_nil_expr(e))
      return true;

    if (
      is_symbol2t(e) && is_shared(to_symbol2t(e)) &&
      !is_partial_order_type(e->type))
      return reject("a global that is not a scalar or an array of scalars");

    bool ok = true;
    e->foreach_operand([this, &ok](const expr2tc &op) {
      if (ok)
        ok = check_expr(op);
    });
    return ok;
  }

  bool is_shared(const symbol2t &sym) const
  {
    return is_partial_order_global(ns, sym.thename);
  }

  bool is_shared_address(const expr2tc &e) const
  {
    if (!is_address_of2t(e))
      return false;
    expr2tc obj = to_address_of2t(e).ptr_obj;
    while (is_index2t(obj))
      obj = to_index2t(obj).source_value;
    return is_symbol2t(obj) && is_shared(to_symbol2t(obj));
  }

  bool reads_shared(const expr2tc &e) const
  {
    if (is_symbol2t(e) && is_shared(to_symbol2t(e)))
      return true;
    bool found = false;
    e->foreach_operand(
      [&](const expr2tc &op) { found = found || reads_shared(op); });
    return found;
  }
};
} // namespace

bool is_partial_order_type(const type2tc &t)
{
  if (is_array_type(t))
    return is_partial_order_type(to_array_type(t).subtype);
  return is_bv_type(t) || is_bool_type(t) || is_fixedbv_type(t) ||
         is_floatbv_type(t);
}

bool is_partial_order_global(const namespacet &ns, const irep_idt &name)
{
  const std::string id = name.as_string();
  if (
    id.find("__ESBMC") != std::string::npos ||
    id.find("pthread_lib") != std::string::npos)
    return false;
  const symbolt *symbol = ns.lookup(name);
  return symbol && symbol->static_lifetime && !symbol->get_type().is_code() &&
         !file_operations::is_bundled_source(
           symbol->location.get_file().as_string());
}

bool partial_order_in_scope(
  const goto_functionst &goto_functions,
  const namespacet &ns,
  bool verifying)
{
  return scope_checkt(ns, false).check(goto_functions) &&
         (!verifying || scope_checkt(ns, true).check(goto_functions));
}
