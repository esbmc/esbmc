#include <goto-programs/partial_order_scope.h>
#include <irep2/irep2_utils.h>
#include <util/base/filesystem.h>
#include <util/message/message.h>

#include <algorithm>
#include <map>
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
    // ... and with it.
    "pthread_join_switch",
    "pthread_mutex_lock_check",
    "pthread_mutex_unlock_check",
    "pthread_mutex_destroy",
    // Signals change nothing a wait observes, since a wait may wake at will.
    "pthread_cond_init",
    "pthread_cond_destroy",
    "pthread_cond_signal",
    "pthread_cond_broadcast",
    "pthread_cond_wait_nocheck",
    "pthread_exit",
    "__ESBMC_atomic_begin",
    "__ESBMC_atomic_end",
    "__ESBMC_really_atomic_begin",
    "__ESBMC_really_atomic_end",
    "__ESBMC_terminate_thread",
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
    "__ESBMC_switch_to_monitor",
    "__ESBMC_switch_from_monitor",
    "__ESBMC_register_monitor",
    "__ESBMC_kill_monitor"};
  return scheduler_intrinsics.count(name) != 0;
}

/// The operational model's thread counters, which symex renames sequentially
/// rather than as events, so a user read would see one thread's view only.
std::string scheduler_state_name(const symbol2t &sym)
{
  static const std::set<std::string> counters{
    "__ESBMC_num_total_threads",
    "__ESBMC_num_threads_running",
    "__ESBMC_blocked_threads_count"};
  const std::string name = sym.thename.as_string();
  const std::string bare = name.starts_with("c:@") ? name.substr(3) : name;
  return counters.count(bare) != 0 ? bare : "";
}

bool is_mutex_type(const type2tc &t)
{
  const std::string name =
    is_struct_type(t)  ? to_struct_type(t).name.as_string()
    : is_union_type(t) ? to_union_type(t).name.as_string()
                       : "";
  return name.find("pthread_mutex_t") != std::string::npos;
}

expr2tc strip_casts(expr2tc e)
{
  while (is_typecast2t(e))
    e = to_typecast2t(e).from;
  return e;
}

bool is_null(const expr2tc &e)
{
  const expr2tc v = strip_casts(e);
  return is_constant(v) && !is_constant_string2t(v);
}

bool is_shared_pointer(const namespacet &ns, const expr2tc &e)
{
  return is_symbol2t(e) && is_pointer_type(e->type) &&
         is_partial_order_global(ns, to_symbol2t(e).thename);
}

/// Calls \p f on each assignment to a shared global pointer, initialisers
/// included, with the instruction's location.
template <typename F>
void each_pointer_assignment(
  const goto_functionst &goto_functions,
  const namespacet &ns,
  F f)
{
  for (const auto &[name, fn] : goto_functions.function_map)
    for (const auto &instruction : fn.body.instructions)
    {
      if (!instruction.is_assign())
        continue;
      const code_assign2t &assign = to_code_assign2t(instruction.code);
      const expr2tc target = strip_casts(assign.target);
      if (is_shared_pointer(ns, target))
        f(to_symbol2t(target).thename, assign.source, instruction.location);
    }
}

class scope_checkt
{
public:
  scope_checkt(const namespacet &ns, bool verifying, bool races, bool deadlocks)
    : ns(ns),
      rejects_assumptions(verifying && deadlocks),
      races(races),
      deadlocks(deadlocks)
  {
  }

  bool check(const goto_functionst &goto_functions)
  {
    functions = &goto_functions;
    find_foreign_pointer_values(goto_functions);
    for (const auto &[name, fn] : goto_functions.function_map)
    {
      if (!is_user_function(name.as_string(), fn))
        continue;
      for (const auto &instruction : fn.body.instructions)
      {
        where = &instruction.location;
        if (!check_instruction(instruction))
          return false;
      }
    }
    return true;
  }

private:
  const namespacet &ns;
  const bool rejects_assumptions;
  const bool races;
  const bool deadlocks;
  const locationt *where = nullptr;
  const goto_functionst *functions = nullptr;
  /// The first assignment of each shared global pointer whose value is not
  /// a global's address or null.
  std::map<irep_idt, const locationt *> foreign_pointer_values;

  static bool
  is_user_function(const std::string &name, const goto_functiont &fn)
  {
    return fn.body_available && !fn.body.hide &&
           name.find("__ESBMC") == std::string::npos &&
           !fn.body.instructions.empty() &&
           !file_operations::is_bundled_source(
             fn.body.instructions.front().location.get_file().as_string());
  }

  bool reject(const std::string &what, const locationt *at = nullptr) const
  {
    log_error(
      "--partial-order does not support {} yet, at {}",
      what,
      (at ? at : where)->as_string());
    return false;
  }

  bool check_instruction(const goto_programt::instructiont &instruction) const
  {
    if (instruction.is_function_call())
      return check_call(to_code_function_call2t(instruction.code));
    if (rejects_assumptions && instruction.is_assume())
      return reject("an assumption under --deadlock-check");
    return check_expr(instruction.code) && check_expr(instruction.guard);
  }

  bool check_call(const code_function_call2t &call) const
  {
    const std::string name = callee_name(call);
    if (!is_symbol2t(call.function))
      return reject("a call through a function pointer");
    if (is_unsupported_call(name))
      return reject("a call to " + name);
    if (rejects_assumptions && name == "__VERIFIER_assume")
      return reject("a call to " + name + " under --deadlock-check");
    if (!check_deadlock_call(name))
      return false;

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

  /// The deadlock claim takes main's end for the process's end.
  bool check_deadlock_call(const std::string &name) const
  {
    if (
      deadlocks &&
      (name == "pthread_exit" || name == "__ESBMC_terminate_thread"))
      return reject("checking deadlocks with a call to " + name);
    return true;
  }

  bool has_body(const code_function_call2t &call) const
  {
    auto it = functions->function_map.find(to_symbol2t(call.function).thename);
    return it != functions->function_map.end() && it->second.body_available;
  }

  /// A lock event names its mutex by the global it is.
  bool check_mutex_arg(
    const std::string &name,
    const code_function_call2t &call) const
  {
    const bool wait = name.starts_with("pthread_cond_wait");
    if (
      !wait && !name.starts_with("pthread_mutex_lock") &&
      !name.starts_with("pthread_mutex_unlock"))
      return true;
    const expr2tc &m = call.operands.at(wait ? 1 : 0);
    if (is_address_of2t(m) && !is_shared_address(m))
      return reject("a mutex that is not a global");
    return true;
  }

  bool check_thread_args(
    const std::string &name,
    const code_function_call2t &call) const
  {
    const bool create = name == "pthread_create";
    const bool join = name.starts_with("pthread_join");
    if (!check_mutex_arg(name, call))
      return false;
    if (create && !is_value_arg(call.operands.at(3)))
      return reject("a thread argument that may point to a local");
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

  /// The thread receives the argument's value at the spawn. A pointer to a
  /// local would let it read the local's value at the end of its owner's run.
  bool is_value_arg(expr2tc arg) const
  {
    while (is_typecast2t(arg))
      arg = to_typecast2t(arg).from;
    return is_null(arg) || is_shared_address(arg) || !has_pointer(arg);
  }

  static bool has_pointer(const expr2tc &e)
  {
    if (is_pointer_type(e->type) || is_address_of2t(e))
      return true;
    bool found = false;
    e->foreach_operand(
      [&found](const expr2tc &op) { found = found || has_pointer(op); });
    return found;
  }

  bool check_expr(const expr2tc &e) const
  {
    if (is_nil_expr(e))
      return true;

    // Symex resolves the pointer to the global when it is locked.
    if (
      is_address_of2t(e) && is_symbol2t(to_address_of2t(e).ptr_obj) &&
      is_mutex_type(to_address_of2t(e).ptr_obj->type))
      return true;

    // A write through it would escape the pointer's value set.
    if (is_address_of2t(e) && is_shared_pointer(ns, root_of(e)))
      return reject("taking the address of a global pointer");

    if (is_symbol2t(e) && !check_symbol(e))
      return false;

    bool ok = true;
    e->foreach_operand([this, &ok](const expr2tc &op) {
      if (ok)
        ok = check_expr(op);
    });
    return ok;
  }

  bool check_symbol(const expr2tc &e) const
  {
    const symbol2t &sym = to_symbol2t(e);
    if (const std::string state = scheduler_state_name(sym); !state.empty())
      return reject("the scheduler state " + state);

    if (!is_shared(sym))
      return true;
    if (!is_partial_order_global_type(e->type))
      return reject(
        "a global that is not a scalar or an array or struct of them");
    if (
      auto it = foreign_pointer_values.find(sym.thename);
      it != foreign_pointer_values.end())
      return reject(
        "a global pointer assigned other than a global's address", it->second);
    // An array or a struct is one object, so accesses to distinct elements or
    // fields would race.
    if (races && (is_array_type(e->type) || is_struct_type(e->type)))
      return reject("checking data races on a global array or struct");
    return true;
  }

  bool is_shared(const symbol2t &sym) const
  {
    return is_partial_order_global(ns, sym.thename);
  }

  static expr2tc root_of(const expr2tc &address)
  {
    expr2tc obj = to_address_of2t(address).ptr_obj;
    while (is_index2t(obj) || is_member2t(obj))
      obj = is_index2t(obj) ? to_index2t(obj).source_value
                            : to_member2t(obj).source_value;
    return obj;
  }

  bool is_shared_address(const expr2tc &e) const
  {
    if (!is_address_of2t(e))
      return false;
    const expr2tc obj = root_of(e);
    return is_symbol2t(obj) && is_shared(to_symbol2t(obj));
  }

  /// A read of a global pointer may see any thread's write, so symex takes
  /// its value set from every value it is assigned, which must name globals.
  void find_foreign_pointer_values(const goto_functionst &goto_functions)
  {
    each_pointer_assignment(
      goto_functions,
      ns,
      [this](const irep_idt &name, const expr2tc &value, const locationt &loc) {
        const expr2tc v = strip_casts(value);
        if (!is_null(v) && !is_shared_address(v))
          foreign_pointer_values.emplace(name, &loc);
      });
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
  // The operational model's sync objects are ordered by LOCK and UNLOCK
  // events, not by their fields.
  if (is_struct_type(t))
  {
    const struct_type2t &s = to_struct_type(t);
    return s.name.as_string().find("pthread_") == std::string::npos &&
           std::all_of(
             s.members.begin(), s.members.end(), [](const type2tc &m) {
               return is_partial_order_type(m);
             });
  }
  return is_bv_type(t) || is_bool_type(t) || is_fixedbv_type(t) ||
         is_floatbv_type(t);
}

bool is_partial_order_global_type(const type2tc &t)
{
  return is_pointer_type(t) || is_partial_order_type(t);
}

std::map<irep_idt, std::vector<expr2tc>> partial_order_pointer_values(
  const goto_functionst &goto_functions,
  const namespacet &ns)
{
  std::map<irep_idt, std::vector<expr2tc>> values;
  each_pointer_assignment(
    goto_functions,
    ns,
    [&values](const irep_idt &name, const expr2tc &value, const locationt &) {
      values[name].push_back(value);
    });
  return values;
}

bool is_partial_order_global(const namespacet &ns, const irep_idt &name)
{
  const std::string id = name.as_string();
  if (
    id.find("__ESBMC") != std::string::npos ||
    id.find("pthread_lib") != std::string::npos)
    return false;
  const symbolt *symbol = ns.lookup(name);
  return symbol && symbol->static_lifetime && !symbol->is_thread_local &&
         !symbol->get_type().is_code() &&
         !file_operations::is_bundled_source(
           symbol->location.get_file().as_string());
}

bool partial_order_in_scope(
  const goto_functionst &goto_functions,
  const namespacet &ns,
  bool verifying,
  bool races,
  bool deadlocks)
{
  return scope_checkt(ns, false, races, deadlocks).check(goto_functions) &&
         (!verifying ||
          scope_checkt(ns, true, races, deadlocks).check(goto_functions));
}
