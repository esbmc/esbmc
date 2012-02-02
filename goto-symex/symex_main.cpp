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
#include "symex_target_equation.h"

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
  statet &state,
  unsigned node_id) {

  total_claims++;

  exprt expr = claim_expr;
  state.rename(expr, ns, node_id);

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

bool
goto_symext::restore_from_dfs_state(void *_dfs)
{
  std::vector<reachability_treet::dfs_position::dfs_state>::const_iterator it;
  unsigned int i;

  const reachability_treet::dfs_position *foo = (const reachability_treet::dfs_position*)_dfs;
  const reachability_treet::dfs_position &dfs = *foo;
  // Symex repeatedly until context switch points. At each point, verify that it
  // happened where we expected it to, and then switch to the correct thread for
  // the history we've been provided with.
  for (it = dfs.states.begin(), i = 0; it != dfs.states.end(); it++, i++) {

    art1->at_end_of_run = false;

    while (!art1->is_at_end_of_run()) {
      // Restore the DFS exploration space so that when an interleaving occurs
      // we take the option leading to the thread we desire to run. This
      // assumes that the DFS exploration path algorithm never changes.
      // Has to occur here; between generating new threads, ESBMC messes with
      // the dfs state.
      for (int dfspos = 0; dfspos < art1->get_cur_state()._DFS_traversed.size();
           dfspos++)
        art1->get_cur_state()._DFS_traversed[dfspos] = true;
      art1->get_cur_state()._DFS_traversed[it->cur_thread] = false;

      symex_step(art1->goto_functions, *art1);
    }
    art1->get_cur_state()._DFS_traversed = it->explored;

    if (art1->get_cur_state()._threads_state.size() != it->num_threads) {
      std::cerr << "Unexpected number of threads when reexploring checkpoint"
                << std::endl;
      abort();
    }

    art1->switch_to_next_execution_state();

    // check we're on the right thread; except on the last run, where there are
    // no more threads to be run.
    if (i + 1 < dfs.states.size())
      assert(art1->get_cur_state().get_active_state_number() == it->cur_thread);

#if 0
// XXX jmorse: can't quite get these sequence numbers to line up when they're
// replayed.
    if (art1->get_cur_state().get_active_state().source.pc->location_number !=
        it->location_number) {
      std::cerr << "Interleave at unexpected location when restoring checkpoint"
                << std::endl;
      abort();
    }
#endif
  }

  return false;
}

void goto_symext::save_checkpoint(const std::string fname) const
{

  reachability_treet::dfs_position pos(*art1);
  if (pos.write_to_file(fname))
    std::cerr << "Couldn't save checkpoint; continuing" << std::endl;

  return;
}

/*******************************************************************\

Function: goto_symext::operator()

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::operator()(const goto_functionst &goto_functions)
{

  reachability_treet art(goto_functions, ns, options);

  int total_states = 0;
  while (art.has_more_states())
  {
    total_states++;
    art.get_cur_state().execute_guard(ns, *target);
    while (!art.is_at_end_of_run())
    {
      symex_step(goto_functions, art);
    }

    art.go_next_state();
  }
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

  merge_gotos(state, ex_state, ex_state.node_id);

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
            if(instruction.function == "c::main")
            {
                ex_state.end_thread(ns, *target);
                ex_state.reexecute_instruction = false;
                art.generate_states_base(exprt());
                art.set_is_at_end_of_run();
            }
            else
            {
                symex_end_of_function(state);
                state.source.pc++;
            }
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
            dereference(tmp, state, false, ex_state.node_id);

            if(!tmp.is_nil() && !options.get_bool_option("deadlock-check"))
            {
              if(ex_state._threads_state.size() > 1)
                if (art.generate_states_before_read(tmp))
                  return;
            }

            symex_goto(art.get_cur_state().get_active_state(), ex_state, ex_state.node_id);
        }
            break;
        case ASSUME:
            if (!state.guard.is_false()) {
                exprt tmp(instruction.guard);
                replace_dynamic_allocation(state, tmp);
                replace_nondet(tmp, ex_state);
                dereference(tmp, state, false, ex_state.node_id);

                exprt tmp1 = tmp;
                state.rename(tmp, ns,ex_state.node_id);

                do_simplify(tmp);
                if (!tmp.is_true())
                {
                  if(ex_state._threads_state.size() > 1)
                    if (art.generate_states_before_read(tmp1))
                      return;

                    exprt tmp2 = tmp;
                    state.guard.guard_expr(tmp2);
                    target->assumption(state.guard, tmp2, state.source);

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

                    replace_dynamic_allocation(state, tmp);
                    replace_nondet(tmp, ex_state);
                    dereference(tmp, state, false, ex_state.node_id);

                    if(ex_state._threads_state.size() > 1)
                      if (art.generate_states_before_read(tmp))
                        return;

                    claim(tmp, msg, state, ex_state.node_id);
                }
            }
            state.source.pc++;
            break;

        case RETURN:
        	 if(!state.guard.is_false())
                         symex_return(state, ex_state, ex_state.node_id);

            state.source.pc++;
            break;

        case ASSIGN:
            if (!state.guard.is_false()) {
                codet deref_code=instruction.code;
                replace_dynamic_allocation(state, deref_code);
                replace_nondet(deref_code, ex_state);
                assert(deref_code.operands().size()==2);

                dereference(deref_code.op0(), state, true, ex_state.node_id);
                dereference(deref_code.op1(), state, false, ex_state.node_id);

                symex_assign(state, ex_state, deref_code, ex_state.node_id);

                state.source.pc++;

                if(ex_state._threads_state.size() > 1)
                {
                  if (art.generate_states_before_assign(deref_code, ex_state))
                    return;
                }
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
                    dereference(deref_code.lhs(), state, true,ex_state.node_id);
                }

                dereference(deref_code.function(), state, false, ex_state.node_id);

                if(deref_code.function().identifier() == "c::__ESBMC_yield")
                {
                   state.source.pc++;
                   ex_state.reexecute_instruction = false;
                   art.generate_states();
                   return;
                }

                if (deref_code.function().identifier() == "c::__ESBMC_switch_to")
                {
                  state.source.pc++;
                  ex_state.reexecute_instruction = false;

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
                    dereference(*it, state, false,ex_state.node_id);
                }

                symex_function_call(goto_functions, ex_state, deref_code);

//                ex_state.reexecute_instruction = false;
//                art.generate_states();

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
                    dereference(deref_code, state, false,ex_state.node_id);
                }

                symex_other(goto_functions, state, ex_state,  ex_state.node_id);
            }
            state.source.pc++;
            break;

        case SYNC:
            throw "SYNC not yet implemented";

        case START_THREAD:
        	if (!state.guard.is_false())
        	{
          	  assert(!instruction.targets.empty());
          	  goto_programt::const_targett goto_target=instruction.targets.front();

              state.source.pc++;
              ex_state.add_thread(state);
              ex_state.get_active_state().source.pc = goto_target;

              //ex_state.deadlock_detection(ns,*target);
              ex_state.update_trds_count(ns,*target);
              ex_state.increament_trds_in_run(ns,*target);

              ex_state.generating_new_threads = ex_state._threads_state.size() - 1;
            }
        	else
        	{
              assert(!instruction.targets.empty());
              goto_programt::const_targett goto_target=instruction.targets.front();
              state.source.pc = goto_target;
            }

            ex_state.reexecute_instruction = false;
            art.generate_states();
            art.set_is_at_end_of_run();

            break;
        case END_THREAD:
            ex_state.end_thread(ns, *target);
            ex_state.reexecute_instruction = false;
            art.generate_states();
            break;
        case ATOMIC_BEGIN:
            state.source.pc++;
            ex_state.increment_active_atomic_number();
            break;
        case ATOMIC_END:
            ex_state.decrement_active_atomic_number();
            state.source.pc++;
            ex_state.reexecute_instruction = false;
            art.generate_states();
            break;
        default:
            assert(false);
    }
}
