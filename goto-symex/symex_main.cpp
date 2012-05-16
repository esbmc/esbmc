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
goto_symext::claim(const expr2tc &claim_expr, const std::string &msg) {

  total_claims++;

  expr2tc new_expr = claim_expr;
  cur_state->rename(new_expr);

  // first try simplifier on it
  do_simplify(new_expr);

  if (is_constant_bool2t(new_expr) &&
      to_constant_bool2t(new_expr).constant_value &&
      !options.get_bool_option("all-assertions"))
    return;

  exprt expr = migrate_expr_back(new_expr);
  cur_state->guard.guard_expr(expr);
  migrate_expr(expr, new_expr);

  remaining_claims++;
  expr2tc guard;
  migrate_expr(cur_state->guard.as_expr(), guard);
  target->assertion(guard, new_expr, msg, cur_state->gen_stack_trace(),
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
    expr2tc tmp;
    migrate_expr(instruction.guard, tmp);
    replace_dynamic_allocation(tmp);
    replace_nondet(tmp);

    dereference(tmp, false);

    exprt even_tmper = migrate_expr_back(tmp);

    symex_goto(even_tmper);
  }
  break;

  case ASSUME:
    if (!cur_state->guard.is_false()) {
      expr2tc tmp;
      migrate_expr(instruction.guard, tmp);
      replace_dynamic_allocation(tmp);
      replace_nondet(tmp);

      dereference(tmp, false);

      cur_state->rename(tmp);
      do_simplify(tmp);

      if (!is_constant_bool2t(tmp) || !to_constant_bool2t(tmp).constant_value) {
	exprt tmp2 = migrate_expr_back(tmp);
        exprt tmp3 = tmp2;
	cur_state->guard.guard_expr(tmp2);

	assume(tmp2);

	// we also add it to the state guard
	cur_state->guard.add(tmp3);
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

        expr2tc tmp;
        migrate_expr(instruction.guard, tmp);
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
      if (make_return_assignment(assign, code)) {
        expr2tc newassign;
        migrate_expr(assign, newassign);
	goto_symext::symex_assign(newassign);
      }
      symex_return();
    }

    cur_state->source.pc++;
    break;

  case ASSIGN:
    if (!cur_state->guard.is_false()) {
      expr2tc deref_code;
      migrate_expr(instruction.code, deref_code);
      replace_dynamic_allocation(deref_code);
      replace_nondet(deref_code);

      code_assign2t &assign = to_code_assign2t(deref_code); 

      dereference(assign.target, true);
      dereference(assign.source, false);

      symex_assign(deref_code);
    }
    cur_state->source.pc++;
    break;

  case FUNCTION_CALL:
    if (!cur_state->guard.is_false()) {
      expr2tc deref_code;
      migrate_expr(instruction.code, deref_code);
      replace_dynamic_allocation(deref_code);
      replace_nondet(deref_code);

      code_function_call2t &call = to_code_function_call2t(deref_code);

      if (!is_nil_expr(call.ret)) {
	dereference(call.ret, true);
      }

      for (std::vector<expr2tc>::iterator it = call.operands.begin();
           it != call.operands.end(); it++)
        if (!is_nil_expr(*it))
          dereference(*it, false);

      exprt tmp = migrate_expr_back(deref_code);
      codet &tmp1 = static_cast<codet&>(tmp);
      code_function_callt new_deref_code = to_code_function_call(tmp1);

      if (has_prefix(new_deref_code.function().identifier().as_string(),
                     "c::__ESBMC")) {
	cur_state->source.pc++;
        expr2tc deref_code2;
        migrate_expr(new_deref_code, deref_code2);
        const code_function_call2t &call = to_code_function_call2t(deref_code2);
	run_intrinsic(call, art, to_symbol2t(call.function).name.as_string());
	return;
      }

      symex_function_call(new_deref_code);
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
goto_symext::run_intrinsic(const code_function_call2t &func_call,
                           reachability_treet &art, const std::string symname)
{

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
    intrinsic_spawn_thread(func_call, art);
  } else if (symname == "c::__ESBMC_terminate_thread") {
    intrinsic_terminate_thread(art);
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
