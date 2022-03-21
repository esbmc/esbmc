/*******************************************************************\

   Module: Symbolic Execution

   Author: Daniel Kroening, kroening@kroening.com Lucas Cordeiro,
     lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <cassert>
#include <goto-symex/execution_state.h>
#include <goto-symex/goto_symex.h>
#include <goto-symex/goto_symex_state.h>
#include <goto-symex/reachability_tree.h>
#include <goto-symex/symex_target_equation.h>

#include <util/c_types.h>
#include <util/config.h>
#include <util/expr_util.h>
#include <irep2/irep2.h>
#include <util/migrate.h>
#include <util/prefix.h>
#include <util/pretty.h>
#include <util/simplify_expr.h>
#include <util/std_expr.h>
#include <vector>

bool goto_symext::check_incremental(const expr2tc &expr, const std::string &msg)
{
  auto rte = std::dynamic_pointer_cast<runtime_encoded_equationt>(target);
  equality2tc question(gen_true_expr(), expr);
  try
  {
    // check whether the assertion holds
    tvt res = rte->ask_solver_question(question);
    // we don't add this assertion to the resulting logical formula
    if(res.is_true())
      // incremental verification succeeded
      return true;
    // this assertion evaluates to false via incremental SMT solving
    if(res.is_false())
    {
      // check assertion to produce a counterexample
      assertion(gen_false_expr(), msg);
      // eliminate subsequent execution paths
      assume(gen_false_expr());
      // incremental verification succeeded
      return true;
    }
    this->msg.status("Incremental verification returned unknown");
    // incremental verification returned unknown
    return false;
  }
  catch(runtime_encoded_equationt::dual_unsat_exception &e)
  {
    this->msg.error(
      "This solver was unable to check this expression. Please try it with "
      "another solver");
  }
  return false;
}

void goto_symext::claim(const expr2tc &claim_expr, const std::string &msg)
{
  // Convert asserts in assumes, if it's not the last loop iteration
  // also, don't convert assertions added by the bidirectional search
  if(inductive_step && first_loop && !cur_state->source.pc->inductive_assertion)
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

  if(options.get_bool_option("smt-symex-assert"))
  {
    if(check_incremental(new_expr, msg))
      // incremental verification has succeeded
      return;
  }

  // add assertion to the target equation
  assertion(new_expr, msg);
}

void goto_symext::assertion(
  const expr2tc &the_assertion,
  const std::string &msg)
{
  expr2tc expr = the_assertion;
  cur_state->guard.guard_expr(expr);
  cur_state->global_guard.guard_expr(expr);
  remaining_claims++;
  target->assertion(
    cur_state->guard.as_expr(),
    expr,
    msg,
    cur_state->gen_stack_trace(),
    cur_state->source,
    first_loop);
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
  target->assumption(tmp_guard, assumption, cur_state->source, first_loop);

  // If we're assuming false, make the guard for the following statement false
  if(is_false(the_assumption))
    cur_state->guard.make_false();
}

std::shared_ptr<goto_symext::symex_resultt> goto_symext::get_symex_result()
{
  return std::shared_ptr<goto_symext::symex_resultt>(
    new goto_symext::symex_resultt(target, total_claims, remaining_claims));
}

void goto_symext::symex_step(reachability_treet &art)
{
  assert(!cur_state->call_stack.empty());

  const goto_programt::instructiont &instruction = *cur_state->source.pc;

  // depth exceeded?
  {
    if(depth_limit != 0 && cur_state->num_instructions > depth_limit)
      cur_state->guard.add(gen_false_expr());
    cur_state->num_instructions++;
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

      symex_return(thecode);
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
      if(has_prefix(id.as_string(), "c:@F@__ESBMC"))
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

  case DECL:
    if(!cur_state->guard.is_false())
      symex_decl(instruction.code);
    cur_state->source.pc++;
    break;

  case DEAD:
    if(!cur_state->guard.is_false())
      symex_dead(instruction.code);
    cur_state->source.pc++;
    break;

  case OTHER:
    if(!cur_state->guard.is_false())
      symex_other(instruction.code);
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
    std::ostringstream oss;
    oss << "GOTO instruction type " << instruction.type;
    oss << " not handled in goto_symext::symex_step"
        << "\n";
    msg.error(oss.str());
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
  if(symname == "c:@F@__ESBMC_yield")
  {
    intrinsic_yield(art);
  }
  else if(symname == "c:@F@__ESBMC_switch_to")
  {
    intrinsic_switch_to(func_call, art);
  }
  else if(symname == "c:@F@__ESBMC_switch_away_from")
  {
    intrinsic_switch_from(art);
  }
  else if(symname == "c:@F@__ESBMC_get_thread_id")
  {
    intrinsic_get_thread_id(func_call, art);
  }
  else if(symname == "c:@F@__ESBMC_set_thread_internal_data")
  {
    intrinsic_set_thread_data(func_call, art);
  }
  else if(symname == "c:@F@__ESBMC_get_thread_internal_data")
  {
    intrinsic_get_thread_data(func_call, art);
  }
  else if(symname == "c:@F@__ESBMC_spawn_thread")
  {
    intrinsic_spawn_thread(func_call, art);
  }
  else if(symname == "c:@F@__ESBMC_terminate_thread")
  {
    intrinsic_terminate_thread(art);
  }
  else if(symname == "c:@F@__ESBMC_get_thread_state")
  {
    intrinsic_get_thread_state(func_call, art);
  }
  else if(symname == "c:@F@__ESBMC_really_atomic_begin")
  {
    intrinsic_really_atomic_begin(art);
  }
  else if(symname == "c:@F@__ESBMC_really_atomic_end")
  {
    intrinsic_really_atomic_end(art);
  }
  else if(symname == "c:@F@__ESBMC_switch_to_monitor")
  {
    intrinsic_switch_to_monitor(art);
  }
  else if(symname == "c:@F@__ESBMC_switch_from_monitor")
  {
    intrinsic_switch_from_monitor(art);
  }
  else if(symname == "c:@F@__ESBMC_register_monitor")
  {
    intrinsic_register_monitor(func_call, art);
  }
  else if(symname == "c:@F@__ESBMC_kill_monitor")
  {
    intrinsic_kill_monitor(art);
  }
  else if(symname == "c:@F@__ESBMC_memset")
  {
    intrinsic_memset(art, func_call);
  }
  else if(symname == "c:@F@__ESBMC_get_object_size")
  {
    intrinsic_get_object_size(func_call, art);
  }
  else if(has_prefix(symname, "c:@F@__ESBMC_overflow"))
  {
    bool is_mult = has_prefix(symname, "c:@F@__ESBMC_overflow_smul") ||
                   has_prefix(symname, "c:@F@__ESBMC_overflow_umul");
    bool is_add = has_prefix(symname, "c:@F@__ESBMC_overflow_sadd") ||
                  has_prefix(symname, "c:@F@__ESBMC_overflow_uadd");
    bool is_sub = has_prefix(symname, "c:@F@__ESBMC_overflow_ssub") ||
                  has_prefix(symname, "c:@F@__ESBMC_overflow_usub");

    assert(func_call.operands.size() == 3);

    const auto &func_type = to_code_type(func_call.function->type);
    assert(func_type.arguments[0] == func_type.arguments[1]);
    assert(is_pointer_type(func_type.arguments[2]));

    expr2tc op;
    if(is_mult)
      op = mul2tc(
        func_type.arguments[0], func_call.operands[0], func_call.operands[1]);
    else if(is_add)
      op = add2tc(
        func_type.arguments[0], func_call.operands[0], func_call.operands[1]);
    else if(is_sub)
      op = sub2tc(
        func_type.arguments[0], func_call.operands[0], func_call.operands[1]);
    else
    {
      assert(0 && "Unknown overflow intrinsics");
    }

    // Assign result of the two arguments to the dereferenced third argument
    symex_assign(code_assign2tc(
      dereference2tc(
        to_pointer_type(func_call.operands[2]->type).subtype,
        func_call.operands[2]),
      op));

    // Perform overflow check and assign it to the return object
    symex_assign(code_assign2tc(func_call.ret, expr2tc(new overflow2t(op))));
  }
  else if(has_prefix(symname, "c:@F@__ESBMC_atomic_load"))
  {
    assert(
      func_call.operands.size() == 3 && "Wrong __ESBMC_atomic_load signature");
    auto &ex_state = art.get_cur_state();
    if(ex_state.cur_state->guard.is_false())
      return;

    expr2tc ptr = func_call.operands[0];
    expr2tc ret = func_call.operands[1];

    symex_assign(code_assign2tc(
      dereference2tc(to_pointer_type(ret->type).subtype, ret),
      dereference2tc(to_pointer_type(ptr->type).subtype, ptr)));
  }
  else if(has_prefix(symname, "c:@F@__ESBMC_atomic_store"))
  {
    assert(
      func_call.operands.size() == 3 && "Wrong __ESBMC_atomic_store signature");
    auto &ex_state = art.get_cur_state();
    if(ex_state.cur_state->guard.is_false())
      return;

    expr2tc ptr = func_call.operands[0];
    expr2tc ret = func_call.operands[1];

    symex_assign(code_assign2tc(
      dereference2tc(to_pointer_type(ptr->type).subtype, ptr),
      dereference2tc(to_pointer_type(ret->type).subtype, ret)));
  }
  else if(has_prefix(symname, "c:@F@__ESBMC_is_little_endian"))
  {
    expr2tc is_little_endian =
      (config.ansi_c.endianess == configt::ansi_ct::IS_LITTLE_ENDIAN)
        ? gen_true_expr()
        : gen_false_expr();
    symex_assign(code_assign2tc(func_call.ret, is_little_endian));
  }
  else if(symname == "c:@F@__ESBMC_builtin_constant_p")
  {
    assert(
      func_call.operands.size() == 1 &&
      "Wrong __ESBMC_builtin_constant_p signature");
    auto &ex_state = art.get_cur_state();
    if(ex_state.cur_state->guard.is_false())
      return;

    expr2tc op1 = func_call.operands[0];
    cur_state->rename(op1);
    symex_assign(code_assign2tc(
      func_call.ret,
      is_constant_int2t(op1) ? gen_one(int_type2()) : gen_zero(int_type2())));
  }
  else if(has_prefix(symname, "c:@F@__ESBMC_sync_fetch_and_add"))
  {
    // Already modelled in builtin_libs
    return;
  }
  else if(has_prefix(symname, "c:@F@__ESBMC_init_var"))
  {
    assert(
      func_call.operands.size() == 1 && "Wrong __ESBMC_init_var signature");
    auto &ex_state = art.get_cur_state();
    if(ex_state.cur_state->guard.is_false())
      return;

    // Get the argument
    expr2tc arg0 = func_call.operands[0];
    internal_deref_items.clear();
    dereference2tc deref(get_empty_type(), arg0);
    dereference(deref, dereferencet::INTERNAL);

    for(const auto &item : internal_deref_items)
    {
      assert(
        item.object->expr_id == expr2t::expr_ids::symbol_id &&
        "__ESBMC_init_var only works for variables");

      // Get the length of the type. This will propagate an exception for dynamic/infinite
      // sized arrays (as expected)
      try
      {
        type_byte_size(item.object->type).to_int64();
      }
      catch(array_type2t::dyn_sized_array_excp *e)
      {
        msg.error("__ESBMC_init_var does not support VLAs");
        abort();
      }
      catch(array_type2t::inf_sized_array_excp *e)
      {
        msg.error("__ESBMC_init_var does not support infinite-length arrays");
        abort();
      }
      expr2tc val = sideeffect2tc(
        item.object->type,
        expr2tc(),
        expr2tc(),
        std::vector<expr2tc>(),
        type2tc(),
        sideeffect2t::nondet);

      symex_assign(code_assign2tc(item.object, val), false, cur_state->guard);
    }

    return;
  }
  else
  {
    std::ostringstream oss;
    oss << "Function call to non-intrinsic prefixed with __ESBMC";
    oss << " (fatal)\nThe name in question: " << symname;
    oss << "\n(NB: the C spec reserves the __ prefix for the compiler"
           " and environment)\n";

    msg.error(oss.str());
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
      "dereference failure: forgotten memory: " + get_pretty_name(it.name),
      cur_state->gen_stack_trace(),
      cur_state->source,
      first_loop);

    total_claims++;
    remaining_claims++;
  }
}
