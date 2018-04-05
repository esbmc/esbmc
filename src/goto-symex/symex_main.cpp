/*******************************************************************\

   Module: Symbolic Execution

   Author: Daniel Kroening, kroening@kroening.com Lucas Cordeiro,
     lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <boost/shared_ptr.hpp>
#include <cassert>
#include <goto-symex/execution_state.h>
#include <goto-symex/goto_symex.h>
#include <goto-symex/goto_symex_state.h>
#include <goto-symex/reachability_tree.h>
#include <goto-symex/symex_target_equation.h>
#include <iostream>
#include <util/c_types.h>
#include <util/config.h>
#include <util/expr_util.h>
#include <util/irep2.h>
#include <util/migrate.h>
#include <util/prefix.h>
#include <util/simplify_expr.h>
#include <util/std_expr.h>
#include <vector>

void goto_symext::claim(const expr2tc &claim_expr, const std::string &msg)
{
  if(inductive_step && first_loop)
  {
    BigInt unwind = cur_state->loop_iterations[first_loop];
    if(unwind < (max_unwind - 1))
    {
      assume(claim_expr);
      return;
    }
  }

  // Can happen when evaluating certain special intrinsics. Gulp.
  if(cur_state->guard.is_false())
    return;

  total_claims++;

  expr2tc new_expr = claim_expr;
  cur_state->rename(new_expr);

  // first try simplifier on it
  do_simplify(new_expr);

  if(is_true(new_expr))
    return;

  cur_state->guard.guard_expr(new_expr);
  cur_state->global_guard.guard_expr(new_expr);
  remaining_claims++;
  target->assertion(
    cur_state->guard.as_expr(),
    new_expr,
    msg,
    cur_state->gen_stack_trace(),
    cur_state->source);
}

void goto_symext::assume(const expr2tc &the_assumption)
{
  expr2tc assumption = the_assumption;
  cur_state->rename(assumption);
  do_simplify(assumption);

  if(is_true(assumption))
    return;

  cur_state->guard.guard_expr(assumption);

  // Irritatingly, assumption destroys its expr argument
  expr2tc tmp_guard = cur_state->guard.as_expr();
  target->assumption(tmp_guard, assumption, cur_state->source);

  // If we're assuming false, make the guard for the following statement false
  if(is_false(the_assumption))
    cur_state->guard.make_false();
}

boost::shared_ptr<goto_symext::symex_resultt> goto_symext::get_symex_result()
{
  return boost::shared_ptr<goto_symext::symex_resultt>(
    new goto_symext::symex_resultt(target, total_claims, remaining_claims));
}

void goto_symext::symex_step(reachability_treet &art)
{
  assert(!cur_state->call_stack.empty());

  const goto_programt::instructiont &instruction = *cur_state->source.pc;

  // depth exceeded?
  {
    if(depth_limit != 0 && cur_state->depth > depth_limit)
      cur_state->guard.add(gen_false_expr());
    cur_state->depth++;
  }

  // Remember the first loop we're entering
  if(inductive_step && instruction.loop_number && !first_loop)
    first_loop = instruction.loop_number;

  // actually do instruction
  switch(instruction.type)
  {
  case SKIP:
  case LOCATION:
    // really ignore
    cur_state->source.pc++;
    break;

  case END_FUNCTION:
    symex_end_of_function();

    // Potentially skip to run another function ptr target; if not,
    // continue
    if(!run_next_function_ptr_target(false))
      cur_state->source.pc++;
    break;

  case GOTO:
  {
    expr2tc tmp(instruction.guard);
    replace_nondet(tmp);

    dereference(tmp, dereferencet::READ);
    replace_dynamic_allocation(tmp);

    symex_goto(tmp);
  }
  break;

  case ASSUME:
    symex_assume();
    cur_state->source.pc++;
    break;

  case ASSERT:
    symex_assert();
    cur_state->source.pc++;
    break;

  case RETURN:
    if(!cur_state->guard.is_false())
    {
      expr2tc thecode = instruction.code, assign;
      if(make_return_assignment(assign, thecode))
      {
        goto_symext::symex_assign(assign);
      }

      symex_return();
    }

    cur_state->source.pc++;
    break;

  case ASSIGN:
    if(!cur_state->guard.is_false())
    {
      code_assign2tc deref_code = instruction.code;

      // XXX jmorse -- this is not fully symbolic.
      if(thrown_obj_map.find(cur_state->source.pc) != thrown_obj_map.end())
      {
        symbol2tc thrown_obj = thrown_obj_map[cur_state->source.pc];

        if(
          is_pointer_type(deref_code->target->type) &&
          !is_pointer_type(thrown_obj->type))
        {
          expr2tc new_thrown_obj(
            new address_of2t(thrown_obj->type, thrown_obj));
          deref_code->source = new_thrown_obj;
        }
        else
          deref_code->source = thrown_obj;

        thrown_obj_map.erase(cur_state->source.pc);
      }

      replace_nondet(deref_code);

      code_assign2t &assign = to_code_assign2t(deref_code);

      dereference(assign.target, dereferencet::WRITE);
      dereference(assign.source, dereferencet::READ);
      replace_dynamic_allocation(deref_code);

      symex_assign(deref_code);
    }

    cur_state->source.pc++;
    break;

  case FUNCTION_CALL:
  {
    expr2tc deref_code = instruction.code;
    replace_nondet(deref_code);

    code_function_call2t &call = to_code_function_call2t(deref_code);

    if(!is_nil_expr(call.ret))
    {
      dereference(call.ret, dereferencet::WRITE);
    }

    replace_dynamic_allocation(deref_code);

    for(auto &operand : call.operands)
      if(!is_nil_expr(operand))
        dereference(operand, dereferencet::READ);

    // Always run intrinsics, whether guard is false or not. This is due to the
    // unfortunate circumstance where a thread starts with false guard due to
    // decision taken in another thread in this trace. In that case the
    // terminate intrinsic _has_ to run, or we explode.
    if(is_symbol2t(call.function))
    {
      const irep_idt &id = to_symbol2t(call.function).thename;
      if(has_prefix(id.as_string(), "__ESBMC"))
      {
        cur_state->source.pc++;
        run_intrinsic(call, art, id.as_string());
        return;
      }
    }

    // Don't run a function call if the guard is false.
    if(!cur_state->guard.is_false())
    {
      symex_function_call(deref_code);
    }
    else
    {
      cur_state->source.pc++;
    }
  }
  break;

  case OTHER:
    if(!cur_state->guard.is_false())
    {
      symex_other();
    }
    cur_state->source.pc++;
    break;

  case CATCH:
    symex_catch();
    break;

  case THROW:
    if(!cur_state->guard.is_false())
    {
      if(symex_throw())
        cur_state->source.pc++;
    }
    else
    {
      cur_state->source.pc++;
    }
    break;

  case THROW_DECL:
    symex_throw_decl();
    cur_state->source.pc++;
    break;

  case THROW_DECL_END:
    // When we reach THROW_DECL_END, we must clear any throw_decl
    if(stack_catch.size())
    {
      // Get to the correct try (always the last one)
      goto_symex_statet::exceptiont *except = &stack_catch.top();

      except->has_throw_decl = false;
      except->throw_list_set.clear();
    }

    cur_state->source.pc++;
    break;

  default:
    std::cerr << "GOTO instruction type " << instruction.type;
    std::cerr << " not handled in goto_symext::symex_step" << std::endl;
    abort();
  }
}

void goto_symext::symex_assume()
{
  if(cur_state->guard.is_false())
    return;

  expr2tc cond = cur_state->source.pc->guard;

  replace_nondet(cond);
  dereference(cond, dereferencet::READ);
  replace_dynamic_allocation(cond);

  assume(cond);
}

void goto_symext::symex_assert()
{
  if(cur_state->guard.is_false())
    return;

  // Don't convert if it's an user provided assertion and we're running in
  // no assertion mode or forward condition
  if(cur_state->source.pc->location.user_provided() && no_assertions)
    return;

  std::string msg = cur_state->source.pc->location.comment().as_string();
  if(msg == "")
    msg = "assertion";

  const goto_programt::instructiont &instruction = *cur_state->source.pc;

  expr2tc tmp = instruction.guard;
  replace_nondet(tmp);

  dereference(tmp, dereferencet::READ);
  replace_dynamic_allocation(tmp);

  claim(tmp, msg);
}

void goto_symext::run_intrinsic(
  const code_function_call2t &func_call,
  reachability_treet &art,
  const std::string &symname)
{
  if(symname == "__ESBMC_yield")
  {
    intrinsic_yield(art);
  }
  else if(symname == "__ESBMC_switch_to")
  {
    intrinsic_switch_to(func_call, art);
  }
  else if(symname == "__ESBMC_switch_away_from")
  {
    intrinsic_switch_from(art);
  }
  else if(symname == "__ESBMC_get_thread_id")
  {
    intrinsic_get_thread_id(func_call, art);
  }
  else if(symname == "__ESBMC_set_thread_internal_data")
  {
    intrinsic_set_thread_data(func_call, art);
  }
  else if(symname == "__ESBMC_get_thread_internal_data")
  {
    intrinsic_get_thread_data(func_call, art);
  }
  else if(symname == "__ESBMC_spawn_thread")
  {
    intrinsic_spawn_thread(func_call, art);
  }
  else if(symname == "__ESBMC_terminate_thread")
  {
    intrinsic_terminate_thread(art);
  }
  else if(symname == "__ESBMC_get_thread_state")
  {
    intrinsic_get_thread_state(func_call, art);
  }
  else if(symname == "__ESBMC_really_atomic_begin")
  {
    intrinsic_really_atomic_begin(art);
  }
  else if(symname == "__ESBMC_really_atomic_end")
  {
    intrinsic_really_atomic_end(art);
  }
  else if(symname == "__ESBMC_switch_to_monitor")
  {
    intrinsic_switch_to_monitor(art);
  }
  else if(symname == "__ESBMC_switch_from_monitor")
  {
    intrinsic_switch_from_monitor(art);
  }
  else if(symname == "__ESBMC_register_monitor")
  {
    intrinsic_register_monitor(func_call, art);
  }
  else if(symname == "__ESBMC_kill_monitor")
  {
    intrinsic_kill_monitor(art);
  }
  else if(symname == "__ESBMC_memset")
  {
    intrinsic_memset(art, func_call);
  }
  else
  {
    std::cerr << "Function call to non-intrinsic prefixed with __ESBMC (fatal)";
    std::cerr << std::endl << "The name in question: " << symname << std::endl;
    std::cerr << "(NB: the C spec reserves the __ prefix for the compiler and "
                 "environment)"
              << std::endl;
    abort();
  }
}

void goto_symext::finish_formula()
{
  if(!memory_leak_check)
    return;

  for(auto const &it : dynamic_memory)
  {
    // Don't check memory leak if the object is automatically deallocated
    if(it.auto_deallocd)
      continue;

    // Assert that the allocated object was freed.
    deallocated_obj2tc deallocd(it.obj);

    equality2tc eq(deallocd, gen_true_expr());
    replace_dynamic_allocation(eq);
    it.alloc_guard.guard_expr(eq);
    cur_state->rename(eq);
    target->assertion(
      it.alloc_guard.as_expr(),
      eq,
      "dereference failure: forgotten memory: " + it.name,
      cur_state->gen_stack_trace(),
      cur_state->source);

    total_claims++;
    remaining_claims++;
  }
}
