#include <ac_config.h>

#include <esbmc/esbmc_parseoptions.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/expr_util.h>
#include <util/irep.h>
#include <util/message.h>
#include <util/prefix.h>
#include <util/std_code.h>
#include <util/symbol.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

void esbmc_parseoptionst::add_property_monitors(
  goto_functionst &goto_functions,
  namespacet &ns [[maybe_unused]])
{
  std::map<std::string, std::pair<std::set<std::string>, expr2tc>> monitors;

  context.foreach_operand([this, &monitors](const symbolt &s) {
    if (
      !has_prefix(s.name, "__ESBMC_property_") ||
      s.name.as_string().find("$type") != std::string::npos)
      return;

    // strip prefix "__ESBMC_property_"
    std::string prop_name = s.name.as_string().substr(17);
    std::set<std::string> used_syms;
    expr2tc main_expr = calculate_a_property_monitor(prop_name, used_syms);
    monitors[prop_name] = std::pair{used_syms, main_expr};
  });

  if (monitors.size() == 0)
    return;

  Forall_goto_functions (f_it, goto_functions)
  {
    /* do not instrument global entry function */
    if (f_it->first == "__ESBMC_main")
      continue;

    /* do also not instrument functions computing the propositions themselves */
    if (has_prefix(f_it->first, "c:@F@") && has_suffix(f_it->first, "_status"))
    {
      const std::string &name = f_it->first.as_string();
      std::string prop_name = name.substr(5, name.length() - 5 - 7);
      if (monitors.find(prop_name) != monitors.end())
        continue;
    }

    log_debug("ltl", "adding monitor exprs in function {}", f_it->first);
    goto_functiont &func = f_it->second;
    goto_programt &prog = func.body;
    Forall_goto_program_instructions (p_it, prog)
      add_monitor_exprs(p_it, prog.instructions, monitors);
  }

  // Find main function; find first function call; insert updates to each
  // property expression. This makes sure that there isn't inconsistent
  // initialization of each monitor boolean.
  goto_functionst::function_mapt::iterator f_it =
    goto_functions.function_map.find("__ESBMC_main");
  assert(f_it != goto_functions.function_map.end());
  std::string main_suffix = "@" + (config.main.empty() ? "main" : config.main);
  const symbol2t *entry_sym = nullptr;
  Forall_goto_program_instructions (p_it, f_it->second.body)
  {
    /* Find the call to the entry point, usually 'main'. At that point
     * everything like pthreads, etc., is already set up. */
    if (p_it->type != FUNCTION_CALL)
      continue;
    const code_function_call2t &func_call = to_code_function_call2t(p_it->code);
    if (!is_symbol2t(func_call.function))
      continue;
    const symbol2t &func_sym = to_symbol2t(func_call.function);
    if (!has_suffix(func_sym.thename, main_suffix))
      continue;

    /* found it */
    entry_sym = &func_sym;
    break;
  }
  assert(entry_sym);

  f_it = goto_functions.function_map.find(entry_sym->thename.as_string());
  assert(f_it != goto_functions.function_map.end());

  goto_programt &body = f_it->second.body;
  goto_programt::instructionst &insn_list = body.instructions;

  /* insert a call to start the monitor thread and after it also to kill it */
  goto_programt::instructiont new_insn;
  new_insn.function = entry_sym->thename;

  expr2tc func_sym = symbol2tc(get_empty_type(), "c:@F@ltl2ba_start_monitor");
  std::vector<expr2tc> args;
  new_insn.make_function_call(code_function_call2tc(expr2tc(), func_sym, args));
  insn_list.insert(insn_list.begin(), new_insn);

  func_sym = symbol2tc(get_empty_type(), "c:@F@ltl2ba_finish_monitor");
  new_insn.make_function_call(code_function_call2tc(expr2tc(), func_sym, args));
  // add this call before each 'return' instruction
  for (auto it = insn_list.begin(); it != insn_list.end(); ++it)
  {
    if (it->type != RETURN)
      continue;
    insn_list.insert(it, new_insn);
  }
}

static void collect_symbol_names(
  const expr2tc &e,
  const std::string &prefix,
  std::set<std::string> &used_syms)
{
  if (is_symbol2t(e))
  {
    const symbol2t &thesym = to_symbol2t(e);
    assert(thesym.rlevel == symbol_renaming_level::level0);
    std::string sym = thesym.get_symbol_name();

    used_syms.insert(sym);
  }
  else
  {
    e->foreach_operand([&prefix, &used_syms](const expr2tc &e) {
      if (!is_nil_expr(e))
        collect_symbol_names(e, prefix, used_syms);
    });
  }
}

expr2tc esbmc_parseoptionst::calculate_a_property_monitor(
  const std::string &name,
  std::set<std::string> &used_syms) const
{
  const symbolt *fn = context.find_symbol("c:@F@" + name + "_status");
  assert(fn);

  const codet &fn_code = to_code(fn->get_value());
  assert(fn_code.get_statement() == "block");
  assert(fn_code.operands().size() == 1);

  const codet &fn_ret = to_code(fn_code.op0());
  assert(fn_ret.get_statement() == "return");
  assert(fn_ret.operands().size() == 1);

  expr2tc new_main_expr;
  migrate_expr(fn_ret.op0(), new_main_expr);

  collect_symbol_names(new_main_expr, name, used_syms);

  return new_main_expr;
}

void esbmc_parseoptionst::add_monitor_exprs(
  goto_programt::targett insn,
  goto_programt::instructionst &insn_list,
  const std::map<std::string, std::pair<std::set<std::string>, expr2tc>>
    &monitors)
{
  // We've been handed an instruction; look for assignments to a symbol
  // referenced by some monitor proposition. When we find one, wrap the
  // assignment in an atomic block so the monitor thread observes it as a
  // single transition. (The explicit context switch to the monitor thread
  // after the assignment — __ESBMC_switch_to_monitor — is currently
  // disabled; see the #if 0 block below.)

  if (!insn->is_assign())
    return;

  code_assign2t &assign = to_code_assign2t(insn->code);

  // Don't allow propositions about things like the contents of an array and
  // suchlike.
  if (!is_symbol2t(assign.target))
    return;

  symbol2t &sym = to_symbol2t(assign.target);

  // Is this actually an assignment that we're interested in?
  const std::string sym_name = sym.get_symbol_name();
  const bool monitored = std::any_of(
    monitors.begin(), monitors.end(), [&sym_name](const auto &monitor) {
      return monitor.second.first.count(sym_name) != 0;
    });

  if (!monitored)
    return;

  goto_programt::instructiont new_insn;

  new_insn.type = ATOMIC_BEGIN;
  new_insn.function = insn->function;
  insn_list.insert(insn, new_insn);

  insn++;

#if 0
  new_insn.type = FUNCTION_CALL;
  expr2tc func_sym =
    symbol2tc(get_empty_type(), "c:@F@__ESBMC_switch_to_monitor");
  std::vector<expr2tc> args;
  new_insn.code = code_function_call2tc(expr2tc(), func_sym, args);
  new_insn.function = insn->function;
  insn_list.insert(insn, new_insn);
#endif

  new_insn.type = ATOMIC_END;
  new_insn.function = insn->function;
  insn_list.insert(insn, new_insn);
}

static unsigned int calc_globals_used(const namespacet &ns, const expr2tc &expr)
{
  if (is_nil_expr(expr))
    return 0;

  if (!is_symbol2t(expr))
  {
    unsigned int globals = 0;

    expr->foreach_operand([&globals, &ns](const expr2tc &e) {
      globals += calc_globals_used(ns, e);
    });
    return globals;
  }

  std::string identifier = to_symbol2t(expr).get_symbol_name();

  if (
    identifier == "NULL" || identifier == "__ESBMC_alloc" ||
    identifier == "__ESBMC_alloc_size")
    return 0;

  const symbolt *sym = ns.lookup(identifier);
  assert(sym);
  if (sym->static_lifetime || sym->get_type().is_dynamic_set())
    return 1;

  return 0;
}

void esbmc_parseoptionst::print_ileave_points(
  namespacet &ns,
  goto_functionst &goto_functions)
{
  forall_goto_functions (fit, goto_functions)
    forall_goto_program_instructions (pit, fit->second.body)
    {
      bool print_insn = false;

      switch (pit->type)
      {
      case GOTO:
      case ASSUME:
      case ASSERT:
      case ASSIGN:
        if (calc_globals_used(ns, pit->guard) > 0)
          print_insn = true;
        break;
      case FUNCTION_CALL:
      {
        const code_function_call2t &deref_code =
          to_code_function_call2t(pit->code);
        if (
          is_symbol2t(deref_code.function) &&
          to_symbol2t(deref_code.function).get_symbol_name() ==
            "c:@F@__ESBMC_yield")
          print_insn = true;
        break;
      }
      case NO_INSTRUCTION_TYPE:
      case OTHER:
      case SKIP:
      case LOCATION:
      case END_FUNCTION:
      case ATOMIC_BEGIN:
      case ATOMIC_END:
      case RETURN:
      case DECL:
      case DEAD:
      case THROW:
      case CATCH:
      case LOOP_INVARIANT:
        break;
      }

      if (print_insn)
        pit->output_instruction(ns, pit->function, std::cout);
    }
}
