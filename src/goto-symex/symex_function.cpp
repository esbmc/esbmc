#include <algorithm>
#include <cassert>
#include <goto-symex/execution_state.h>
#include <goto-symex/goto_symex.h>
#include <langapi/language_util.h>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/cprover_prefix.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/prefix.h>
#include <util/pretty.h>
#include <util/std_expr.h>

bool goto_symext::get_unwind_recursion(
  const irep_idt &identifier,
  BigInt unwind)
{
  BigInt this_loop_max_unwind = max_unwind;

  if(unwind != 0)
  {
    if(options.get_bool_option("abort-on-recursion"))
      abort();

    if(
      (k_induction || inductive_step) &&
      !options.get_bool_option("disable-inductive-step"))
    {
      log_warning(
        "k-induction does not support recursion yet. Disabling inductive step");

      // Disable inductive step on recursion
      options.set_option("disable-inductive-step", true);
    }

    const symbolt &symbol = *ns.lookup(identifier);

    std::string msg = "Unwinding recursion " + id2string(symbol.name) +
                      " iteration " + integer2string(unwind);

    if(this_loop_max_unwind != 0)
      msg += " (" + integer2string(this_loop_max_unwind) + " max)";

    log_status("{}", msg);
  }

  return this_loop_max_unwind != 0 && unwind >= this_loop_max_unwind;
}

unsigned goto_symext::argument_assignments(
  const irep_idt &function_identifier,
  const code_type2t &function_type,
  const std::vector<expr2tc> &arguments)
{
  // iterates over the operands
  std::vector<expr2tc>::const_iterator it1 = arguments.begin();

  // iterates over the types of the arguments
  for(unsigned int name_idx = 0; name_idx < function_type.arguments.size();
      ++name_idx)
  {
    // if you run out of actual arguments there was a mismatch
    if(it1 == arguments.end())
    {
      claim(gen_false_expr(), "function call: not enough arguments");
      return UINT_MAX;
    }

    const irep_idt &identifier = function_type.argument_names[name_idx];

    // Don't assign arguments if they have no name, see regression spec21
    if(identifier == "")
      continue;

    if(is_nil_expr(*it1))
    {
      ; // XXX jmorse, is this valid?
    }
    else
    {
      expr2tc rhs = *it1;

      // it should be the same exact type
      auto const &arg_type = function_type.arguments[name_idx];
      if(!base_type_eq(arg_type, rhs->type, ns))
      {
        const type2tc &f_arg_type = arg_type;
        const type2tc &f_rhs_type = rhs->type;

        // we are willing to do some limited conversion
        if(
          (is_number_type(f_arg_type) || is_pointer_type(f_arg_type)) &&
          (is_number_type(f_rhs_type) || is_pointer_type(f_rhs_type)))
        {
          rhs = typecast2tc(arg_type, rhs);
        }
        else
        {
          log_error(
            "function call: argument \"{}\" type mismatch: got {}, expected {}",
            id2string(identifier),
            get_type_id((*it1)->type),
            get_type_id(arg_type));
          abort();
        }
      }

      // 'Declare' the argument before assigning a value to it
      symbol2tc lhs(function_type.arguments[name_idx], identifier);
      symex_decl(code_decl2tc(lhs->type, identifier));

      // Assign value to function argument
      // TODO: Should we hide it (true means hidden)?
      symex_assign(code_assign2tc(lhs, rhs), true);
    }

    it1++;
  }

  unsigned va_index = UINT_MAX;
  if(function_type.ellipsis)
  {
    // These are va_arg arguments; their types may differ from call to call
    unsigned va_count = 0;
    while(new_context.find_symbol(
            id2string(function_identifier) + "::va_arg" +
            std::to_string(va_count)) != nullptr)
      ++va_count;

    va_index = va_count;
    for(; it1 != arguments.end(); it1++, va_count++)
    {
      irep_idt identifier =
        id2string(function_identifier) + "::va_arg" + std::to_string(va_count);

      // add to symbol table
      symbolt symbol;
      symbol.id = identifier;
      symbol.name = "va_arg" + std::to_string(va_count);
      symbol.type = migrate_type_back((*it1)->type);

      if(new_context.move(symbol))
      {
        log_error("Couldn't add new va_arg symbol");
        abort();
      }
      // 'Declare' the argument before assigning a value to it
      symex_decl(code_decl2tc((*it1)->type, identifier));

      symbol2tc sym((*it1)->type, identifier);
      cur_state->top().level1.get_ident_name(sym);

      // Assign value to function argument
      // TODO: Should we hide it (true means hidden)?
      symex_assign(code_assign2tc(sym, *it1), true);
    }
  }
  else if(it1 != arguments.end())
  {
    // we got too many arguments, but we will just ignore them
  }

  return va_index;
}

void goto_symext::symex_function_call(const expr2tc &code)
{
  const code_function_call2t &call = to_code_function_call2t(code);

  if(is_symbol2t(call.function))
    symex_function_call_code(code);
  else
    symex_function_call_deref(code);
}

void goto_symext::symex_function_call_code(const expr2tc &expr)
{
  const code_function_call2t &call = to_code_function_call2t(expr);
  const irep_idt &identifier = to_symbol2t(call.function).thename;

  // find code in function map
  goto_functionst::function_mapt::const_iterator it =
    goto_functions.function_map.find(identifier);

  if(it == goto_functions.function_map.end())
  {
    if(has_prefix(identifier.as_string(), "symex::invalid_object"))
    {
      log_warning("WARNING: function ptr call with no target, ");
      cur_state->source.pc++;
      return;
    }

    log_error(
      "failed to find `{}' in function_map",
      get_pretty_name(identifier.as_string()));
    abort();
  }

  const goto_functiont &goto_function = it->second;

  BigInt &unwinding_counter = cur_state->function_unwind[identifier];

  // see if it's too much
  if(get_unwind_recursion(identifier, unwinding_counter))
  {
    if(!no_unwinding_assertions)
    {
      claim(gen_false_expr(), "recursion unwinding assertion");
    }
    else
    {
      // Add an unwinding assumption.
      expr2tc now_guard = cur_state->guard.as_expr();
      not2tc not_now(now_guard);
      target->assumption(now_guard, not_now, cur_state->source, first_loop);
    }

    cur_state->source.pc++;
    return;
  }

  if(!goto_function.body_available)
  {
    log_warning(
      "no body for function {}", get_pretty_name(identifier.as_string()));

    /* TODO: if it is a C function with no prototype, assert/claim that all
     *       calls to this function have the same number of parameters and that
     *       they - after type promotion - are compatible. */

    if(!is_nil_expr(call.ret))
    {
      unsigned int &nondet_count = get_nondet_counter();
      symbol2tc rhs(
        call.ret->type, "nondet$symex::" + i2string(nondet_count++));

      symex_assign(code_assign2tc(call.ret, rhs));
    }

    cur_state->source.pc++;
    return;
  }

  // read the arguments -- before the locality renaming
  std::vector<expr2tc> arguments = call.operands;
  for(auto &argument : arguments)
    cur_state->rename(argument);

  // Rename the return value to level1, identifying the data object / storage
  // to which the return value should be written. This is important in the case
  // of recursion, in which case the lexical variable (level0) has multiple
  // live instances.
  expr2tc ret_value = call.ret;
  if(!is_nil_expr(ret_value) && !is_empty_type(ret_value->type))
    cur_state->rename_address(ret_value);

  // increase unwinding counter
  ++unwinding_counter;

  // produce a new frame
  assert(!cur_state->call_stack.empty());
  goto_symex_statet::framet &frame =
    cur_state->new_frame(cur_state->source.thread_nr);

  // copy L1 renaming from previous frame
  frame.level1 = cur_state->previous_frame().level1;
  frame.level1.thread_id = cur_state->source.thread_nr;

  frame.calling_location = cur_state->source;
  frame.entry_guard = cur_state->guard;

  // assign arguments
  type2tc tmp_type = migrate_type(goto_function.type);

  frame.va_index =
    argument_assignments(identifier, to_code_type(tmp_type), arguments);

  frame.end_of_function = --goto_function.body.instructions.end();
  frame.return_value = ret_value;
  frame.function_identifier = identifier;
  frame.hidden = goto_function.body.hide;

  cur_state->source.is_set = true;
  cur_state->source.pc = goto_function.body.instructions.begin();
  cur_state->source.prog = &goto_function.body;
}

static std::list<std::pair<guardt, symbol2tc>>
get_function_list(const expr2tc &expr)
{
  std::list<std::pair<guardt, symbol2tc>> l;

  if(is_if2t(expr))
  {
    std::list<std::pair<guardt, symbol2tc>> l1, l2;
    const if2t &ifexpr = to_if2t(expr);
    expr2tc guardexpr = ifexpr.cond;
    not2tc notguardexpr(guardexpr);

    // Get sub items, them iterate over adding the relevant guard
    l1 = get_function_list(ifexpr.true_value);
    for(auto &it : l1)
      it.first.add(guardexpr);

    l2 = get_function_list(ifexpr.false_value);
    for(auto &it : l2)
      it.first.add(notguardexpr);

    l1.splice(l1.begin(), l2);
    return l1;
  }

  if(is_symbol2t(expr))
  {
    guardt guard;
    guard.make_true();
    std::pair<guardt, symbol2tc> p(guard, symbol2tc(expr));
    l.push_back(p);
    return l;
  }

  if(is_typecast2t(expr))
    return get_function_list(to_typecast2t(expr).from);

  log_error(
    "Unexpected irep id {} in function ptr dereference", get_expr_id(expr));
  // So, the function may point at something invalid. If that's the case,
  // wait for a solve-time pointer validity assertion to detect that. Return
  // nothing to call right now.
  return l;
}

void goto_symext::symex_function_call_deref(const expr2tc &expr)
{
  const code_function_call2t &call = to_code_function_call2t(expr);
  assert(cur_state->top().cur_function_ptr_targets.size() == 0);

  // Indirect function call. The value is dereferenced, so we'll get either an
  // address_of a symbol, or a set of if ireps. For symbols we'll invoke
  // symex_function_call_symbol, when dealing with if's we need to fork and
  // merge.
  if(is_nil_expr(call.function))
  {
    log_error(
      "Function pointer call with no targets; irep: {}", call.pretty(0));
    abort();
  }

  // Generate a list of functions to call. We'll then proceed to call them,
  // and will later on merge them.
  expr2tc func_ptr = call.function;
  dereference(func_ptr, dereferencet::READ);

  // Match the two varieties of failed symbol we can encounter,
  if(
    is_symbol2t(func_ptr) &&
    (has_prefix(
       to_symbol2t(func_ptr).thename.as_string(), "symex::invalid_object") ||
     to_symbol2t(func_ptr).thename.as_string().find("$object") !=
       std::string::npos))
  {
    // Emit warning; perform no function call behaviour. Increment PC
    // XXX jmorse - no location information any more.
    log_status(
      "No target candidate for function call {}",
      from_expr(ns, "", call.function));
    cur_state->source.pc++;
    return;
  }

  std::list<std::pair<guardt, symbol2tc>> l = get_function_list(func_ptr);

  /* Internal check that all symbols are actually of 'code' type (modulo the
   * guard) */
  for(const auto &elem : l)
  {
    const guardt &guard = elem.first;
    const symbol2tc &sym = elem.second;
    if(!guard.is_false() && !is_code_type(sym))
    {
      bool known_internal_error = guard.is_true();
      if(known_internal_error)
      {
        log_error(
          "non-code call target '{}' generated at {}",
          sym->thename.as_string());
        abort();
      }
      log_debug(
        "non-code call target '{}' generated at {}", sym->thename.as_string());
    }
  }

  // Store.
  for(auto &it : l)
  {
    goto_functionst::function_mapt::const_iterator fit =
      goto_functions.function_map.find(it.second->thename);

    const std::string pretty_name = it.second->thename.as_string().substr(
      it.second->thename.as_string().find_last_of('@') + 1);

    if(fit == goto_functions.function_map.end() || !fit->second.body_available)
    {
      log_warning("no body for function {}", pretty_name);

      continue; // XXX, find out why this fires on SV-COMP 14 benchmark
      // 32_7a_cilled_true_linux-3.8-rc1-drivers--ata--pata_legacy.ko-main.cil.out.c
      // Where it probably shouldn't, as that var is defined. Module name
      // difference?
    }

    // Set up a merge of the current state into the target function.
    statet::goto_state_listt &goto_state_list =
      cur_state->top().goto_state_map[fit->second.body.instructions.begin()];

    cur_state->top().cur_function_ptr_targets.emplace_back(
      fit->second.body.instructions.begin(), it.second);

    goto_state_list.emplace_back(*cur_state);
    statet::goto_statet &new_state = goto_state_list.back();
    expr2tc guardexpr = it.first.as_expr();
    cur_state->rename(guardexpr);
    new_state.guard.add(guardexpr);
  }

  cur_state->top().function_ptr_call_loc = cur_state->source.pc;
  cur_state->top().function_ptr_combine_target = cur_state->source.pc;
  cur_state->top().function_ptr_combine_target++;
  cur_state->top().orig_func_ptr_call = expr;

  if(!run_next_function_ptr_target(true))
    cur_state->source.pc++;
}

bool goto_symext::run_next_function_ptr_target(bool first)
{
  if(cur_state->call_stack.empty())
    return false;

  if(cur_state->top().cur_function_ptr_targets.size() == 0)
    return false;

  // Record a merge - when all function ptr target runs are completed, they'll
  // be merged into the state when the instruction after the func call is run.
  // But, don't do it the first time, or we'll have a merge that's effectively
  // unconditional.
  if(!first)
  {
    statet::goto_state_listt &goto_state_list =
      cur_state->top()
        .goto_state_map[cur_state->top().function_ptr_combine_target];
    goto_state_list.emplace_back(*cur_state);
  }

  // Take one function ptr target out of the list and jump to it. A previously
  // recorded merge will ensure it gets the right state.
  std::pair<goto_programt::const_targett, expr2tc> p =
    cur_state->top().cur_function_ptr_targets.front();
  cur_state->top().cur_function_ptr_targets.pop_front();

  goto_programt::const_targett target = p.first;
  expr2tc target_symbol = p.second;

  cur_state->guard.make_false();
  cur_state->source.pc = target;

  // Merge pre-function-ptr-call state in immediately.
  merge_gotos();

  // Now switch back to the original call location so that the call appears
  // to originate from there...
  cur_state->source.pc = cur_state->top().function_ptr_call_loc;

  // And setup the function call.
  code_function_call2tc call = cur_state->top().orig_func_ptr_call;
  call->function = target_symbol;
  goto_symex_statet::framet &cur_frame = cur_state->top();

  if(cur_state->top().cur_function_ptr_targets.size() == 0)
    cur_frame.orig_func_ptr_call = expr2tc();

  symex_function_call_code(call);

  return true;
}

void goto_symext::pop_frame()
{
  assert(!cur_state->call_stack.empty());

  statet::framet &frame = cur_state->top();

  // restore state
  cur_state->source.pc = frame.calling_location.pc;
  cur_state->source.prog = frame.calling_location.prog;

  if(!cur_state->guard.is_false())
    cur_state->guard = frame.entry_guard;

  // clear locals from L2 renaming
  for(auto const &it : frame.local_variables)
  {
    type2tc ptr(new pointer_type2t(pointer_type2()));
    symbol2tc l1_sym(ptr, it.base_name);
    frame.level1.get_ident_name(l1_sym);

    // Call free on alloca'd objects
    if(
      it.base_name.as_string().find("return_value$_alloca") !=
      std::string::npos)
      symex_free(code_free2tc(l1_sym));

    // Erase from level 1 propagation
    cur_state->value_set.erase(l1_sym->get_symbol_name());

    cur_state->level2.remove(it);

    // Construct an l1 name on the fly - this is a temporary hack for when
    // the value set is storing things in a not-an-irep-idt form.
    symbol2tc tmp_expr(
      get_empty_type(), it.base_name, it.lev, it.l1_num, 0, it.t_num, 0);
    cur_state->value_set.erase(to_symbol2t(tmp_expr).get_symbol_name());
  }

  // decrease recursion unwinding counter
  if(!frame.function_identifier.empty())
    --cur_state->function_unwind[frame.function_identifier];

  cur_state->pop_frame();
}

void goto_symext::symex_end_of_function()
{
  pop_frame();
}

bool goto_symext::make_return_assignment(expr2tc &assign, const expr2tc &code)
{
  statet::framet &frame = cur_state->top();
  const code_return2t &ret = to_code_return2t(code);

  if(!is_nil_expr(ret.operand))
  {
    expr2tc value = ret.operand;

    dereference(value, dereferencet::READ);

    if(!is_nil_expr(frame.return_value))
    {
      assign = code_assign2tc(frame.return_value, value);

      if(frame.return_value->type != value->type)
      {
        typecast2tc cast(frame.return_value->type, value);
        assign = code_assign2tc(frame.return_value, cast);
      }

      return true;
    }
  }
  else if(!is_nil_expr(frame.return_value))
  {
    log_error("return with unexpected value");
    abort();
  }

  return false;
}

void goto_symext::symex_return(const expr2tc &code)
{
  // we treat this like an unconditional
  // goto to the end of the function

  // put into state-queue
  statet::goto_state_listt &goto_state_list =
    cur_state->top().goto_state_map[cur_state->top().end_of_function];

  goto_state_list.emplace_back(*cur_state);

  // check whether the stack limit and return
  // value optimization have been activated.
  if(stack_limit > 0 && no_return_value_opt)
  {
    code->foreach_operand([this](const expr2tc &e) {
      // check whether the stack size has been reached.
      claim(
        (cur_state->top().process_stack_size(e, stack_limit)),
        "Stack limit property was violated");
    });
  }

  // kill this one
  cur_state->guard.make_false();
}
