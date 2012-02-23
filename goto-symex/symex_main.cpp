/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com
		Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <assert.h>
#include <iostream>
#include <vector>

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

/*******************************************************************\

Function: goto_symext::claim

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::claim(
  const exprt &claim_expr,
  const std::string &msg,
  statet &state) {

  total_claims++;

  exprt expr = claim_expr;
  state.rename(expr, ns);

  // first try simplifier on it
  if (!expr.is_false())
    do_simplify(expr);

  if (expr.is_true() &&
    !options.get_bool_option("all-assertions"))
  return;

  state.guard.guard_expr(expr);

  remaining_claims++;
  target->assertion(state.guard, expr, msg, state.gen_stack_trace(),
                    state.source);
}

/*******************************************************************\

Function: goto_symext::operator()

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

goto_symext::symex_resultt *
goto_symext::get_symex_result(void)
{

  return new goto_symext::symex_resultt(target, total_claims, remaining_claims);
}

/*******************************************************************\

Function: goto_symext::symex_step

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_step(
        const goto_functionst &goto_functions,
        reachability_treet & art) {

  execution_statet &ex_state = art.get_cur_state();
  statet &state = ex_state.get_active_state();

  assert(!state.call_stack.empty());

  const goto_programt::instructiont &instruction = *state.source.pc;

  if (config.options.get_option("break-at") != "") {
    int insn_num = strtol(config.options.get_option("break-at").c_str(), NULL, 10);
    if (instruction.location_number == insn_num) {
      // If you're developing ESBMC on a machine that isn't x86, I'll send you
      // cookies.
#ifndef _WIN32
      __asm__("int $3");
#else
      std::cerr << "Can't trap on windows, sorry" << std::endl;
      abort();
#endif
    }
  }

  merge_gotos(state, ex_state);

  // depth exceeded?
  {
      unsigned max_depth = atoi(options.get_option("depth").c_str());
      if (max_depth != 0 && state.depth > max_depth)
          state.guard.add(false_exprt());
      state.depth++;
  }

  if (options.get_bool_option("symex-trace")) {
    const goto_programt p_dummy;
    goto_functions_templatet<goto_programt>::function_mapt::const_iterator it =
      goto_functions.function_map.find(instruction.function);

    const goto_programt &p_real = it->second.body;
    const goto_programt &p = (it == goto_functions.function_map.end()) ? p_dummy : p_real;
    p.output_instruction(ns, "", std::cout, state.source.pc, false, false);
  }

    // actually do instruction
    switch (instruction.type) {
        case SKIP:
            // really ignore
            state.source.pc++;
            break;
        case END_FUNCTION:
          symex_end_of_function(state);
          state.source.pc++;
          break;
        case LOCATION:
            target->location(state.guard, state.source);
            state.source.pc++;
            break;
        case GOTO:
        {
            exprt tmp(instruction.guard);
            replace_dynamic_allocation(state, tmp);
            replace_nondet(tmp, ex_state);
            dereference(tmp, state, false);

           symex_goto(art.get_cur_state().get_active_state(), ex_state, tmp);
        }
            break;
        case ASSUME:
            if (!state.guard.is_false()) {
                exprt tmp(instruction.guard);
                replace_dynamic_allocation(state, tmp);
                replace_nondet(tmp, ex_state);
                dereference(tmp, state, false);

                exprt tmp1 = tmp;
                state.rename(tmp, ns);

                do_simplify(tmp);
                if (!tmp.is_true())
                {
                    exprt tmp2 = tmp;
                    state.guard.guard_expr(tmp2);
                    target->assumption(state.guard, tmp2, state.source);

                    // we also add it to the state guard
                    state.guard.add(tmp);

                    state.source.pc++;

                    if(ex_state.threads_state.size() > 1)
                      if (art.generate_states_after_read(tmp1))
                        return;
               } else {
                state.source.pc++;
               }
            } else {
              state.source.pc++;
            }
            break;

        case ASSERT:
            if (!state.guard.is_false()) {
                if (!options.get_bool_option("no-assertions") ||
                        !state.source.pc->location.user_provided()
                        || options.get_bool_option("deadlock-check")) {

                    std::string msg = state.source.pc->location.comment().as_string();
                    if (msg == "") msg = "assertion";
                    exprt tmp(instruction.guard);

                    replace_dynamic_allocation(state, tmp);
                    replace_nondet(tmp, ex_state);
                    dereference(tmp, state, false);

                    state.source.pc++;

                    claim(tmp, msg, state);
               } else {
                 state.source.pc++;
               }
            } else {
              state.source.pc++;
            }
            break;

        case RETURN:
        	 if(!state.guard.is_false()) {
                         const code_returnt &code =
                           to_code_return(instruction.code);
                         code_assignt assign;
                         if (make_return_assignment(state, ex_state, assign, code))
                           goto_symext::symex_assign(state, ex_state, assign);
                         symex_return(state, ex_state);
                 }

            state.source.pc++;
            break;

        case ASSIGN:
            if (!state.guard.is_false()) {
                codet deref_code=instruction.code;
                replace_dynamic_allocation(state, deref_code);
                replace_nondet(deref_code, ex_state);
                assert(deref_code.operands().size()==2);

                dereference(deref_code.op0(), state, true);
                dereference(deref_code.op1(), state, false);

                state.source.pc++;

                symex_assign(state, ex_state, deref_code);
            }
            else
              state.source.pc++;
            break;
        case FUNCTION_CALL:
            if (!state.guard.is_false())
            {
                code_function_callt deref_code =
                        to_code_function_call(instruction.code);

                replace_dynamic_allocation(state, deref_code);
                replace_nondet(deref_code, ex_state);

                if (deref_code.lhs().is_not_nil()) {
                    dereference(deref_code.lhs(), state, true);
                }

                dereference(deref_code.function(), state, false);

                if(deref_code.function().identifier() == "c::__ESBMC_yield")
                {
                   state.source.pc++;
                   art.generate_states();
                   return;
                }

                if (deref_code.function().identifier() == "c::__ESBMC_switch_to")
                {
                  state.source.pc++;

                  assert(deref_code.arguments().size() == 1);

                  // Switch to other thread.
                  exprt &num = deref_code.arguments()[0];
                  if (num.id() != "constant")
                    throw "Can't switch to non-constant thread id no";

                  unsigned int tid = binary2integer(num.value().as_string(), false).to_long();
                  ex_state.set_active_state(tid);
                  return;
                }

                Forall_expr(it, deref_code.arguments()) {
                    dereference(*it, state, false);
                }

                symex_function_call(goto_functions, ex_state, deref_code);
            }
            else
            {
                state.source.pc++;
            }
            break;

        case OTHER:
            if (!state.guard.is_false()) {
                codet deref_code(instruction.code);
                const irep_idt &statement = deref_code.get_statement();
                if (statement == "cpp_delete" ||
                        statement == "cpp_delete[]" ||
                        statement == "free" ||
                        statement == "printf") {
                    replace_dynamic_allocation(state, deref_code);
                    replace_nondet(deref_code, ex_state);
                    dereference(deref_code, state, false);
                }

                symex_other(goto_functions, state, ex_state);
            }
            state.source.pc++;
            break;

        default:
            std::cerr << "GOTO instruction type " << instruction.type;
            std::cerr << " not handled in goto_symext::symex_step" << std::endl;
            abort();
    }
}
