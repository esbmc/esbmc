/*******************************************************************\

   Module: Symbolic Execution

   Author: Daniel Kroening, kroening@kroening.com Lucas Cordeiro,
     lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>
#include <assert.h>
#include <iostream>
#include <vector>

#include <prefix.h>
#include <std_expr.h>
#include <expr_util.h>

#include "goto_symex.h"
#include "goto_symex_state.h"
#include "execution_state.h"
#include "symex_target_equation.h"
#include "reachability_tree.h"

#include <std_expr.h>
#include "../ansi-c/c_types.h"
#include <simplify_expr.h>
#include "config.h"

void
goto_symext::claim(const exprt &claim_expr, const std::string &msg) {

  total_claims++;

  exprt expr = claim_expr;
  expr2tc new_expr;
  migrate_expr(expr, new_expr);
  cur_state->rename(new_expr);
  expr = migrate_expr_back(new_expr);

  // first try simplifier on it
  if (!expr.is_false())
    do_simplify(expr);

  if (expr.is_true() &&
      !options.get_bool_option("all-assertions"))
    return;

  cur_state->guard.guard_expr(expr);

  remaining_claims++;
  expr2tc guard, newexpr;
  migrate_expr(cur_state->guard.as_expr(), guard);
  migrate_expr(expr, newexpr);
  target->assertion(guard, newexpr, msg, cur_state->gen_stack_trace(),
                    cur_state->source);
}

void
goto_symext::assume(const exprt &assumption)
{

  // Irritatingly, assumption destroys its expr argument
  expr2tc guard, assumpt_dup;
  migrate_expr(cur_state->guard.as_expr(), guard);
  migrate_expr(assumption, assumpt_dup);
  target->assumption(guard, assumpt_dup, cur_state->source);
  return;
}

goto_symext::symex_resultt *
goto_symext::get_symex_result(void)
{

  return new goto_symext::symex_resultt(target, total_claims, remaining_claims);
}

void
goto_symext::symex_step(reachability_treet & art)
{

  assert(!cur_state->call_stack.empty());

  const goto_programt::instructiont &instruction = *cur_state->source.pc;

  merge_gotos();

  // depth exceeded?
  {
    unsigned max_depth = atoi(options.get_option("depth").c_str());
    if (max_depth != 0 && cur_state->depth > max_depth)
      cur_state->guard.add(false_exprt());
    cur_state->depth++;
  }

  // actually do instruction
  switch (instruction.type) {
  case SKIP:
  case LOCATION:
    // really ignore
    cur_state->source.pc++;
    break;

  case END_FUNCTION:
    symex_end_of_function();

    // Potentially skip to run another function ptr target; if not,
    // continue
    if (!run_next_function_ptr_target(false))
      cur_state->source.pc++;
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
    if (!cur_state->guard.is_false()) {
      exprt tmp(instruction.guard);
      replace_dynamic_allocation(tmp);
      replace_nondet(tmp);
      dereference(tmp, false);

      exprt tmp1 = tmp;
      expr2tc new_tmp;
      migrate_expr(tmp, new_tmp);
      cur_state->rename(new_tmp);
      tmp = migrate_expr_back(new_tmp);

      do_simplify(tmp);
      if (!tmp.is_true()) {
	exprt tmp2 = tmp;
	cur_state->guard.guard_expr(tmp2);

	assume(tmp2);

	// we also add it to the state guard
	cur_state->guard.add(tmp);
      }
    }
    cur_state->source.pc++;
    break;

  case ASSERT:
    if (!cur_state->guard.is_false()) {
      if (!options.get_bool_option("no-assertions") ||
          !cur_state->source.pc->location.user_provided()
          || options.get_bool_option("deadlock-check")) {

	std::string msg = cur_state->source.pc->location.comment().as_string();
	if (msg == "") msg = "assertion";
	exprt tmp(instruction.guard);

	replace_dynamic_allocation(tmp);
	replace_nondet(tmp);
	dereference(tmp, false);

	claim(tmp, msg);
      }
    }
    cur_state->source.pc++;
    break;

  case RETURN:
    if (!cur_state->guard.is_false()) {
      const code_returnt &code =
        to_code_return(instruction.code);
      code_assignt assign;
      if (make_return_assignment(assign, code))
	goto_symext::symex_assign(assign);
      symex_return();
    }

    cur_state->source.pc++;
    break;

  case ASSIGN:
    if (!cur_state->guard.is_false()) {
      codet deref_code = instruction.code;
      replace_dynamic_allocation(deref_code);
      replace_nondet(deref_code);
      assert(deref_code.operands().size() == 2);

      dereference(deref_code.op0(), true);
      dereference(deref_code.op1(), false);

      symex_assign(deref_code);
    }
    cur_state->source.pc++;
    break;

  case FUNCTION_CALL:
    if (!cur_state->guard.is_false()) {
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
	cur_state->source.pc++;
	run_intrinsic(deref_code, art,
	              deref_code.function().identifier().as_string());
	return;
      }

      symex_function_call(deref_code);
    } else   {
      cur_state->source.pc++;
    }
    break;

  case OTHER:
    if (!cur_state->guard.is_false()) {
      symex_other();
    }
    cur_state->source.pc++;
    break;

  case CATCH:
    symex_catch(*cur_state);
    cur_state->source.pc++;
    break;

  case THROW:
    symex_throw(*cur_state);
    cur_state->source.pc++;
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
  expr2tc call2;
  migrate_expr(call, call2);
  const code_function_call2t &func_call = to_code_function_call2t(call2);

  if (symname == "c::__ESBMC_yield") {
    intrinsic_yield(art);
  } else if (symname == "c::__ESBMC_switch_to") {
    intrinsic_switch_to(func_call, art);
  } else if (symname == "c::__ESBMC_switch_away_from") {
    intrinsic_switch_from(art);
  } else if (symname == "c::__ESBMC_get_thread_id") {
    intrinsic_get_thread_id(func_call, art);
  } else if (symname == "c::__ESBMC_set_thread_internal_data") {
    intrinsic_set_thread_data(func_call, art);
  } else if (symname == "c::__ESBMC_get_thread_internal_data") {
    intrinsic_get_thread_data(func_call, art);
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
