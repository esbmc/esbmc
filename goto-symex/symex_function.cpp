/*******************************************************************\

   Module: Symbolic Execution of ANSI-C

   Author: Daniel Kroening, kroening@kroening.com Lucas Cordeiro,
     lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <assert.h>

#include <expr_util.h>
#include <i2string.h>
#include <cprover_prefix.h>
#include <prefix.h>
#include <arith_tools.h>
#include <base_type.h>
#include <std_expr.h>

#include <ansi-c/c_types.h>

#include "goto_symex.h"
#include "execution_state.h"

bool
goto_symext::get_unwind_recursion(
  const irep_idt &identifier, unsigned unwind)
{
  unsigned long this_loop_max_unwind = max_unwind;

  #if 1
  if (unwind != 0) {
    const symbolt &symbol = ns.lookup(identifier);

    std::string msg =
      "Unwinding recursion " +
      id2string(symbol.display_name()) +
      " iteration " + i2string(unwind);

    if (this_loop_max_unwind != 0)
      msg += " (" + i2string(this_loop_max_unwind) + " max)";

    std::cout << msg << std::endl;
  }
  #endif

  return this_loop_max_unwind != 0 &&
         unwind >= this_loop_max_unwind;
}

void
goto_symext::argument_assignments(
  const code_typet &function_type, statet &state,
  const exprt::operandst &arguments)
{
  // iterates over the operands
  exprt::operandst::const_iterator it1 = arguments.begin();

  // these are the types of the arguments
  const code_typet::argumentst &argument_types = function_type.arguments();

  // iterates over the types of the arguments
  for (code_typet::argumentst::const_iterator
       it2 = argument_types.begin();
       it2 != argument_types.end();
       it2++)
  {
    // if you run out of actual arguments there was a mismatch
    if (it1 == arguments.end())
      throw "function call: not enough arguments";

    const code_typet::argumentt &argument = *it2;

    // this is the type the n-th argument should be
    const typet &arg_type = argument.type();

    const irep_idt &identifier = argument.get_identifier();

    if (identifier == "")
      throw "no identifier for function argument";

    const symbolt &symbol = ns.lookup(identifier);
    exprt lhs = symbol_expr(symbol);

    if (it1->is_nil()) {
    } else   {
      exprt rhs = *it1;

      // it should be the same exact type
      if (!base_type_eq(arg_type, rhs.type(), ns)) {
	const typet &f_arg_type = ns.follow(arg_type);
	const typet &f_rhs_type = ns.follow(rhs.type());

	// we are willing to do some limited conversion
	if ((f_arg_type.id() == typet::t_signedbv ||
	     f_arg_type.id() == typet::t_unsignedbv ||
	     f_arg_type.id() == typet::t_bool ||
	     f_arg_type.id() == typet::t_pointer ||
	     f_arg_type.id() == typet::t_fixedbv) &&
	    (f_rhs_type.id() == typet::t_signedbv ||
	     f_rhs_type.id() == typet::t_unsignedbv ||
	     f_rhs_type.id() == typet::t_bool ||
	     f_rhs_type.id() == typet::t_pointer ||
	     f_rhs_type.id() == typet::t_fixedbv)) {
	  rhs.make_typecast(arg_type);
	} else   {
	  std::string error = "function call: argument \"" +
	                      id2string(identifier) +
	                      "\" type mismatch: got " +
	                      it1->type().to_string() + ", expected " +
	                      arg_type.to_string();
	  throw error;
	}
      }

      guardt guard;
      symex_assign_symbol(state, lhs, rhs, guard);
    }

    it1++;
  }

  if (function_type.has_ellipsis()) {
    for (; it1 != arguments.end(); it1++)
    {
    }
  } else if (it1 != arguments.end())      {
    // we got too many arguments, but we will just ignore them
  }
}

void
goto_symext::symex_function_call(statet &state,
                                 const code_function_callt &code)
{
  const exprt &function = code.function();

  if (function.id() == exprt::symbol)
    symex_function_call_symbol(state, code);
  else
    symex_function_call_deref(state, code);
}

void
goto_symext::symex_function_call_symbol(statet &state,
                                        const code_function_callt &code)
{
  target->location(state.guard, state.source);

  assert(code.function().id() == exprt::symbol);

  symex_function_call_code(state, code);
}

void
goto_symext::symex_function_call_code(statet &state,
                                      const code_function_callt &call)
{
  const irep_idt &identifier =
    to_symbol_expr(call.function()).get_identifier();

  // find code in function map

  goto_functionst::function_mapt::const_iterator it =
    goto_functions.function_map.find(identifier);

  if (it == goto_functions.function_map.end()) {
    if (call.function().invalid_object()) {
      std::cout << "WARNING: function ptr call with no target, ";
      std::cout << call.location() << std::endl;
      state.source.pc++;
      return;
    }

    throw "failed to find `" + id2string(identifier) + "' in function_map";
  }

  const goto_functionst::goto_functiont &goto_function = it->second;

  unsigned &unwinding_counter = state.function_unwind[identifier];

  // see if it's too much
  if (get_unwind_recursion(identifier, unwinding_counter)) {
    if (!options.get_bool_option("no-unwinding-assertions"))
      claim(false_exprt(), "recursion unwinding assertion", state);

    state.source.pc++;
    return;
  }

  if (!goto_function.body_available) {
    if (body_warnings.insert(identifier).second) {
      std::string msg = "**** WARNING: no body for function " + id2string(
        identifier);
      std::cerr << msg << std::endl;
    }

    if (call.lhs().is_not_nil()) {
      unsigned int &nondet_count = get_nondet_counter();
      exprt rhs = exprt("nondet_symbol", call.lhs().type());
      rhs.identifier("symex::" + i2string(nondet_count++));
      rhs.location() = call.location();
      guardt guard;
      symex_assign_rec(state, call.lhs(), rhs, guard);
    }

    state.source.pc++;
    return;
  }

  // read the arguments -- before the locality renaming
  exprt::operandst arguments = call.arguments();
  for (unsigned i = 0; i < arguments.size(); i++)
  {
    state.rename(arguments[i], ns);
  }

  // increase unwinding counter
  unwinding_counter++;

  // produce a new frame
  assert(!state.call_stack.empty());
  goto_symex_statet::framet &frame = state.new_frame(state.source.thread_nr);

  // copy L1 renaming from previous frame
  frame.level1 = state.previous_frame().level1;
  frame.level1._thread_id = state.source.thread_nr;

  unsigned &frame_nr = state.function_frame[identifier];
  frame_nr++;

  frame.calling_location = state.source;

  // preserve locality of local variables
  locality(frame_nr, state, goto_function);

  // assign arguments
  argument_assignments(goto_function.type, state, arguments);

  frame.end_of_function = --goto_function.body.instructions.end();
  frame.return_value = call.lhs();
  frame.function_identifier = identifier;

  state.source.is_set = true;
  state.source.pc = goto_function.body.instructions.begin();
  state.source.prog = &goto_function.body;
}

static std::list<std::pair<guardt, exprt> >
get_function_list(const exprt &expr)
{
  std::list<std::pair<guardt, exprt> > l;

  if (expr.id() == "if") {
    std::list<std::pair<guardt, exprt> > l1, l2;
    exprt guardexpr = expr.op0();
    exprt notguardexpr = not_exprt(guardexpr);

    // Get sub items, them iterate over adding the relevant guard
    l1 = get_function_list(expr.op1());
    for (std::list<std::pair<guardt, exprt> >::iterator it = l1.begin();
         it != l1.end(); it++)
      it->first.add(guardexpr);

    l2 = get_function_list(expr.op2());
    for (std::list<std::pair<guardt, exprt> >::iterator it = l2.begin();
         it != l2.end(); it++)
      it->first.add(notguardexpr);

    l1.splice(l1.begin(), l2);
    return l1;
  } else if (expr.id() == "symbol") {
    guardt guard;
    guard.make_true();
    std::pair<guardt, exprt> p(guard, expr);
    l.push_back(p);
    return l;
  } else {
    std::cerr << "Unexpected irep id " << expr.id() <<
    " in function ptr dereference" << std::endl;
    abort();
  }
}

void
goto_symext::symex_function_call_deref(statet &state,
                                       const code_function_callt &call)
{

  assert(state.top().cur_function_ptr_targets.size() == 0);

  // Indirect function call. The value is dereferenced, so we'll get either an
  // address_of a symbol, or a set of if ireps. For symbols we'll invoke
  // symex_function_call_symbol, when dealing with if's we need to fork and
  // merge.
  if (call.op1().is_nil()) {
    std::cerr << "Function pointer call with no targets; irep: ";
    std::cerr << call.pretty(0) << std::endl;
    abort();
  }

  // Generate a list of functions to call. We'll then proceed to call them,
  // and will later on merge them.
  exprt funcptr = call.op1();
  dereference(funcptr, false);

  if (funcptr.invalid_object()) {
    // Emit warning; perform no function call behaviour. Increment PC
    std::cout << call.op1().location().as_string() << std::endl;
    std::cout << "No target candidate for function call " <<
    from_expr(ns, "", call.op1()) << std::endl;
    state.source.pc++;
    return;
  }

  std::list<std::pair<guardt, exprt> > l = get_function_list(funcptr);

  // Store.
  for (std::list<std::pair<guardt, exprt> >::iterator it = l.begin();
       it != l.end(); it++) {

    goto_functionst::function_mapt::const_iterator fit =
      goto_functions.function_map.find(it->second.identifier());
    if (fit == goto_functions.function_map.end() ||
        !fit->second.body_available) {
      std::cerr << "Couldn't find symbol " << it->second.identifier();
      std::cerr << " or body not available, during function ptr dereference";
      std::cerr << std::endl;
      abort();
    }

    // Set up a merge of the current state into the target function.
    statet::goto_state_listt &goto_state_list =
      state.top().goto_state_map[fit->second.body.instructions.begin()];

    state.top().cur_function_ptr_targets.push_back(
      std::pair<goto_programt::const_targett, exprt>(
        fit->second.body.instructions.begin(),
        it->second)
      );

    goto_state_list.push_back(statet::goto_statet(state));
    statet::goto_statet &new_state = goto_state_list.back();
    exprt guardexpr = it->first.as_expr();
    state.rename(guardexpr, ns);
    new_state.guard.add(guardexpr);
  }

  state.top().function_ptr_call_loc = state.source.pc;
  state.top().function_ptr_combine_target = state.source.pc;
  state.top().function_ptr_combine_target++;
  state.top().orig_func_ptr_call = new code_function_callt(call);

  run_next_function_ptr_target(state, true);
}

bool
goto_symext::run_next_function_ptr_target(statet &state, bool first)
{

  if (state.call_stack.empty())
    return false;

  if (state.top().cur_function_ptr_targets.size() == 0)
    return false;

  // Record a merge - when all function ptr target runs are completed, they'll
  // be merged into the state when the instruction after the func call is run.
  // But, don't do it the first time, or we'll have a merge that's effectively
  // unconditional.
  if (!first) {
    statet::goto_state_listt &goto_state_list =
      state.top().goto_state_map[state.top().function_ptr_combine_target];
    goto_state_list.push_back(statet::goto_statet(state));
  }

  // Take one function ptr target out of the list and jump to it. A previously
  // recorded merge will ensure it gets the right state.
  std::pair<goto_programt::const_targett, exprt> p =
    state.top().cur_function_ptr_targets.front();
  state.top().cur_function_ptr_targets.pop_front();

  goto_programt::const_targett target = p.first;
  exprt target_symbol = p.second;

  state.guard.make_false();
  state.source.pc = target;

  // Merge pre-function-ptr-call state in immediately.
  merge_gotos(state);

  // Now switch back to the original call location so that the call appears
  // to originate from there...
  state.source.pc = state.top().function_ptr_call_loc;

  // And setup the function call.
  code_function_callt call = *state.top().orig_func_ptr_call;
  call.function() = target_symbol;
  goto_symex_statet::framet &cur_frame = state.top();

  if (state.top().cur_function_ptr_targets.size() == 0)
    delete cur_frame.orig_func_ptr_call;

  symex_function_call_code(state, call);


  return true;
}

void
goto_symext::pop_frame(statet &state)
{
  assert(!state.call_stack.empty());

  statet::framet &frame = state.top();

  // restore state
  state.source.pc = frame.calling_location.pc;
  state.source.prog = frame.calling_location.prog;

  // clear locals from L2 renaming
  for (statet::framet::local_variablest::const_iterator
       it = frame.local_variables.begin();
       it != frame.local_variables.end();
       it++)
    state.level2.remove(*it);

  // decrease recursion unwinding counter
  if (frame.function_identifier != "")
    state.function_unwind[frame.function_identifier]--;

  state.pop_frame();
}

void
goto_symext::symex_end_of_function(statet &state)
{
  pop_frame(state);
}

void
goto_symext::locality(
  unsigned frame_nr, statet &state,
  const goto_functionst::goto_functiont &goto_function)
{
  goto_programt::local_variablest local_identifiers;

  for (goto_programt::instructionst::const_iterator
       it = goto_function.body.instructions.begin();
       it != goto_function.body.instructions.end();
       it++)
    local_identifiers.insert(
      it->local_variables.begin(),
      it->local_variables.end());

  statet::framet &frame = state.top();

  for (goto_programt::local_variablest::const_iterator
       it = local_identifiers.begin();
       it != local_identifiers.end();
       it++)
  {
    frame.level1.rename(*it, frame_nr);
    irep_idt l1_name = frame.level1.get_ident_name(*it);
    frame.local_variables.insert(l1_name);
  }
}

bool
goto_symext::make_return_assignment(statet &state, code_assignt &assign,
  const code_returnt &code)
{
  statet::framet &frame = state.top();

  target->location(state.guard, state.source);

  if (code.operands().size() == 1) {
    exprt value(code.op0());

    dereference(value, false);

    if (frame.return_value.is_not_nil()) {
      assign = code_assignt(frame.return_value, value);

      if (assign.lhs().type() != assign.rhs().type())
	assign.rhs().make_typecast(assign.lhs().type());

      //make sure that we assign two expressions of the same type
      assert(assign.lhs().type() == assign.rhs().type());
      return true;
    }
  } else   {
    if (frame.return_value.is_not_nil())
      throw "return with unexpected value";
  }

  return false;
}

void
goto_symext::symex_return(void)
{

  // we treat this like an unconditional
  // goto to the end of the function

  // put into state-queue
  statet::goto_state_listt &goto_state_list =
    cur_state->top().goto_state_map[cur_state->top().end_of_function];

  goto_state_list.push_back(statet::goto_statet(*cur_state));

  // kill this one
  cur_state->guard.make_false();
}
