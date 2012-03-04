/*******************************************************************\

   Module: Symbolic Execution

   Author: Daniel Kroening, kroening@kroening.com Lucas Cordeiro,
     lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <assert.h>
#include <iostream>
#include <vector>

#include <prefix.h>
#include <std_expr.h>
#include <rename.h>
#include <expr_util.h>

#include "goto_symex.h"
#include "goto_symex_state.h"
#include "execution_state.h"
#include "symex_target_equation.h"
#include "reachability_tree.h"

#include <std_expr.h>
#include "../ansi-c/c_types.h"
#include <base_type.h>
#include <simplify_expr.h>
#include "config.h"

void
goto_symext::claim(const exprt &claim_expr, const std::string &msg) {

  total_claims++;

  exprt expr = claim_expr;
  cur_state->rename(expr, ns);

  // first try simplifier on it
  if (!expr.is_false())
    do_simplify(expr);

  if (expr.is_true() &&
      !options.get_bool_option("all-assertions"))
    return;

  cur_state->guard.guard_expr(expr);

  remaining_claims++;
  target->assertion(cur_state->guard, expr, msg, cur_state->gen_stack_trace(),
                    cur_state->source);
}

void
goto_symext::assume(const exprt &assumption)
{

  // Irritatingly, assumption destroys its expr argument
  exprt assumpt_dup = assumption;
  target->assumption(cur_state->guard, assumpt_dup, cur_state->source);
  return;
}

goto_symext::symex_resultt *
goto_symext::get_symex_result(void)
{

  return new goto_symext::symex_resultt(target, total_claims, remaining_claims);
}

void
goto_symext::symex_step(reachability_treet & art) {

  execution_statet &ex_state = art.get_cur_state();
  statet &state = ex_state.get_active_state();

  assert(!state.call_stack.empty());

  const goto_programt::instructiont &instruction = *state.source.pc;

  merge_gotos();

  // depth exceeded?
  {
    unsigned max_depth = atoi(options.get_option("depth").c_str());
    if (max_depth != 0 && state.depth > max_depth)
      state.guard.add(false_exprt());
    state.depth++;
  }

  // actually do instruction
  switch (instruction.type) {
  case SKIP:
    // really ignore
    state.source.pc++;
    break;

  case END_FUNCTION:
    symex_end_of_function(state);

    // Potentially skip to run another function ptr target; if not,
    // continue
    if (!run_next_function_ptr_target(state, false))
      state.source.pc++;
    break;

  case LOCATION:
    target->location(state.guard, state.source);
    state.source.pc++;
    break;

  case GOTO:
  {
    exprt tmp(instruction.guard);
    replace_dynamic_allocation(tmp);
    replace_nondet(tmp);
    dereference(tmp, false);

    symex_goto(tmp);
  }
  break;

  case ASSUME:
    if (!state.guard.is_false()) {
      exprt tmp(instruction.guard);
      replace_dynamic_allocation(tmp);
      replace_nondet(tmp);
      dereference(tmp, false);

      exprt tmp1 = tmp;
      state.rename(tmp, ns);

      do_simplify(tmp);
      if (!tmp.is_true()) {
	exprt tmp2 = tmp;
	state.guard.guard_expr(tmp2);

	assume(tmp2);

	// we also add it to the state guard
	state.guard.add(tmp);
      }
    }
    state.source.pc++;
    break;

  case ASSERT:
    if (!state.guard.is_false()) {
      if (!options.get_bool_option("no-assertions") ||
          !state.source.pc->location.user_provided()
          || options.get_bool_option("deadlock-check")) {

	std::string msg = state.source.pc->location.comment().as_string();
	if (msg == "") msg = "assertion";
	exprt tmp(instruction.guard);

	replace_dynamic_allocation(tmp);
	replace_nondet(tmp);
	dereference(tmp, false);

	claim(tmp, msg);
      }
    }
    state.source.pc++;
    break;

  case RETURN:
    if (!state.guard.is_false()) {
      const code_returnt &code =
        to_code_return(instruction.code);
      code_assignt assign;
      if (make_return_assignment(state, assign, code))
	goto_symext::symex_assign(state, assign);
      symex_return();
    }

    state.source.pc++;
    break;

  case ASSIGN:
    if (!state.guard.is_false()) {
      codet deref_code = instruction.code;
      replace_dynamic_allocation(deref_code);
      replace_nondet(deref_code);
      assert(deref_code.operands().size() == 2);

      dereference(deref_code.op0(), true);
      dereference(deref_code.op1(), false);

      symex_assign(state, deref_code);
    }
    state.source.pc++;
    break;

  case FUNCTION_CALL:
    if (!state.guard.is_false()) {
      code_function_callt deref_code =
        to_code_function_call(instruction.code);

      replace_dynamic_allocation(deref_code);
      replace_nondet(deref_code);

      if (deref_code.lhs().is_not_nil()) {
	dereference(deref_code.lhs(), true);
      }

      Forall_expr(it, deref_code.arguments()) {
	dereference(*it, false);
      }

      if (has_prefix(deref_code.function().identifier().as_string(),
                     "c::__ESBMC")) {
	state.source.pc++;
	run_intrinsic(deref_code, art,
	              deref_code.function().identifier().as_string());
	return;
      }

      symex_function_call(state, deref_code);
    } else   {
      state.source.pc++;
    }
    break;

  case OTHER:
    if (!state.guard.is_false()) {
      symex_other();
    }
    state.source.pc++;
    break;

  default:
    std::cerr << "GOTO instruction type " << instruction.type;
    std::cerr << " not handled in goto_symext::symex_step" << std::endl;
    abort();
  }
}

void
goto_symext::run_intrinsic(code_function_callt &call, reachability_treet &art,
  const std::string symname)
{

  if (symname == "c::__ESBMC_yield") {
    intrinsic_yield(art);
  } else if (symname == "c::__ESBMC_switch_to") {
    intrinsic_switch_to(call, art);
  } else if (symname == "c::__ESBMC_get_thread_id") {
    intrinsic_get_thread_id(call, art);
  } else if (symname == "c::__ESBMC_set_thread_internal_data") {
    intrinsic_set_thread_data(call, art);
  } else if (symname == "c::__ESBMC_get_thread_internal_data") {
    intrinsic_get_thread_data(call, art);
  } else if (symname == "c::__ESBMC_spawn_thread") {
    intrinsic_spawn_thread(call, art);
  } else if (symname == "c::__ESBMC_terminate_thread") {
    intrinsic_terminate_thread(art);
  } else if (symname == "c::__ESBMC_get_thread_state") {
    intrinsic_get_thread_state(call, art);
  } else {
    std::cerr << "Function call to non-intrinsic prefixed with __ESBMC (fatal)";
    std::cerr << std::endl << "The name in question: " << symname << std::endl;
    std::cerr <<
    "(NB: the C spec reserves the __ prefix for the compiler and environment)"
              << std::endl;
    abort();
  }

  return;
}
