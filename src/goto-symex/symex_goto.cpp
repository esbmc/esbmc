/*******************************************************************\

   Module: Symbolic Execution

   Author: Daniel Kroening, kroening@kroening.com Lucas Cordeiro,
     lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <cassert>
#include <fstream>
#include <goto-symex/goto_symex.h>
#include <goto-symex/slice.h>
#include <goto-symex/symex_target_equation.h>
#include <iostream>
#include <langapi/language_ui.h>
#include <solvers/smtlib/smtlib_conv.h>
#include <util/expr_util.h>
#include <util/irep2.h>
#include <util/migrate.h>
#include <util/prefix.h>
#include <util/std_expr.h>

void
goto_symext::symex_goto(const expr2tc &old_guard)
{
  const goto_programt::instructiont &instruction = *cur_state->source.pc;

  expr2tc new_guard = old_guard;
  cur_state->rename(new_guard);
  do_simplify(new_guard);

  bool new_guard_false = false, new_guard_true = false;

  new_guard_false = ((is_false(new_guard)) || cur_state->guard.is_false());
  new_guard_true = is_true(new_guard);

  if (!new_guard_false && options.get_bool_option("smt-symex-guard"))
  {
    auto rte = boost::dynamic_pointer_cast<runtime_encoded_equationt>(target);

    equality2tc question(gen_true_expr(), new_guard);
    try {
      symex_slicet slicer;
      symex_target_equationt eq = dynamic_cast<const symex_target_equationt&>(*target);
      slicer.slice_for_symbols(eq, question);

      tvt res = rte.get()->ask_solver_question(question);

      if (res.is_false())
        new_guard_false = true;
      else if (res.is_true())
        new_guard_true = true;
    } catch (runtime_encoded_equationt::dual_unsat_exception &e) {
      // Assumptions mean that the guard is never satisfiable as true or false,
      // basically means we've assume'd away the possibility of hitting this
      // point.
      new_guard_false = true;
    }
  }

  goto_programt::const_targett goto_target =
    instruction.targets.front();

  bool forward =
    cur_state->source.pc->location_number <
    goto_target->location_number;

  // Check if we are inside a loop, during inductive step
  if(inductive_step && (instruction.loop_number != 0))
  {
    // We just entered the loop, save the loop number
    if(forward)
      loop_numbers.push(instruction.loop_number);
    else
    {
      // We are leaving the loop, remove from stack
      assert(instruction.loop_number == loop_numbers.top());
      loop_numbers.pop();
    }
  }

  statet::framet &frame=cur_state->top();
  if (new_guard_false)
  {
    // reset unwinding counter
    if(instruction.is_backwards_goto())
      frame.loop_iterations[instruction.loop_number] = 0;

    // next instruction
    cur_state->source.pc++;

    return; // nothing to do
  }

  assert(!instruction.targets.empty());

  // we only do deterministic gotos for now
  if (instruction.targets.size() != 1)
    throw "no support for non-deterministic gotos";

  // backwards?
  if (!forward)
  {
    unsigned &unwind = frame.loop_iterations[instruction.loop_number];
    unwind++;

    if (get_unwind(cur_state->source, unwind)) {
      loop_bound_exceeded(new_guard);

      // reset unwinding
      unwind = 0;

      // next instruction
      cur_state->source.pc++;
      return;
    }

    if (new_guard_true) {
      cur_state->source.pc = goto_target;
      return; // nothing else to do
    }
  }

  goto_programt::const_targett new_state_pc, state_pc;

  if (forward) {
    new_state_pc = goto_target; // goto target instruction
    state_pc = cur_state->source.pc;
    state_pc++; // next instruction
  } else {
    new_state_pc = cur_state->source.pc;
    new_state_pc++;
    state_pc = goto_target;
  }

  cur_state->source.pc = state_pc;

  // put into state-queue
  statet::goto_state_listt &goto_state_list =
    cur_state->top().goto_state_map[new_state_pc];

  goto_state_list.push_back(statet::goto_statet(*cur_state));
  statet::goto_statet &new_state = goto_state_list.back();

  // adjust guards
  if (new_guard_true) {
    cur_state->guard.make_false();
  } else {
    // produce new guard symbol
    expr2tc guard_expr;

    if (is_symbol2t(new_guard) ||
        (is_not2t(new_guard) && is_symbol2t(to_not2t(new_guard).value))) {
      guard_expr = new_guard;
    } else {
      guard_expr = guard_identifier();

      expr2tc new_rhs = new_guard;
      new_rhs = not2tc(new_rhs);
      do_simplify(new_rhs);

      cur_state->assignment(guard_expr, new_rhs, false);

      target->assignment(
        gen_true_expr(),
        guard_expr, guard_expr,
        new_rhs,
        cur_state->source,
        cur_state->gen_stack_trace(),
        symex_targett::HIDDEN);

      guard_expr = not2tc(guard_expr);
    }

    not2tc not_guard_expr(guard_expr);
    do_simplify(not_guard_expr);

    if (forward) {
      new_state.guard.add(guard_expr);
      cur_state->guard.add(not_guard_expr);
    } else   {
      cur_state->guard.add(guard_expr);
      new_state.guard.add(not_guard_expr);
    }
  }
}

void
goto_symext::merge_gotos(void)
{
  statet::framet &frame = cur_state->top();

  // first, see if this is a target at all
  statet::goto_state_mapt::iterator state_map_it =
    frame.goto_state_map.find(cur_state->source.pc);

  if (state_map_it == frame.goto_state_map.end())
    return;  // nothing to do

  // we need to merge
  statet::goto_state_listt &state_list = state_map_it->second;

  for (statet::goto_state_listt::reverse_iterator
       list_it = state_list.rbegin();
       list_it != state_list.rend();
       list_it++)
  {
    statet::goto_statet &goto_state = *list_it;

    // do SSA phi functions
    phi_function(goto_state);

    merge_value_sets(goto_state);

    // adjust guard
    cur_state->guard |= goto_state.guard;

    // adjust depth
    cur_state->depth = std::min(cur_state->depth, goto_state.depth);
  }

  // clean up to save some memory
  frame.goto_state_map.erase(state_map_it);
}

void
goto_symext::merge_value_sets(const statet::goto_statet &src)
{
  if (cur_state->guard.is_false()) {
    cur_state->value_set = src.value_set;
    return;
  }

  cur_state->value_set.make_union(src.value_set);
}

void
goto_symext::phi_function(const statet::goto_statet &goto_state)
{
  // go over all variables to see what changed
  std::set<renaming::level2t::name_record> variables;

  goto_state.level2.get_variables(variables);
  cur_state->level2.get_variables(variables);

  guardt tmp_guard;
  if(!variables.empty()
     && !cur_state->guard.is_false()
     && !goto_state.guard.is_false())
  {
    tmp_guard = goto_state.guard;

    // this gets the diff between the guards
    tmp_guard -= cur_state->guard;
  }

  for (std::set<renaming::level2t::name_record>::const_iterator
       it = variables.begin();
       it != variables.end();
       it++)
  {
    if (goto_state.level2.current_number(*it) ==
        cur_state->level2.current_number(*it))
      continue;  // not changed

    if (it->base_name == guard_identifier_s)
      continue;  // just a guard

    if (has_prefix(it->base_name.as_string(),"symex::invalid_object"))
      continue;

    // changed!
    const symbolt &symbol = ns.lookup(it->base_name);

    type2tc type;
    typet old_type = symbol.type;
    migrate_type(symbol.type, type);

    expr2tc rhs;

    if (cur_state->guard.is_false() || goto_state.guard.is_false()) {
      rhs = symbol2tc(type, symbol.name);

      // Try to get the value
      renaming::level2t::rename_to_record(rhs, *it);
      goto_state.level2.rename(rhs);
    } else {
      symbol2tc true_val(type, symbol.name);
      symbol2tc false_val(type, symbol.name);

      // Semi-manually rename these symbols: we may be referring to an l1
      // variable not in the current scope, thus we need to directly specify
      // which l1 variable we're dealing with.
      renaming::level2t::rename_to_record(true_val, *it);
      renaming::level2t::rename_to_record(false_val, *it);

      // Try to get the symbol's value
      goto_state.level2.rename(true_val);
      cur_state->level2.rename(false_val);

      rhs = if2tc(type, tmp_guard.as_expr(), true_val, false_val);
    }

    expr2tc lhs;
    migrate_expr(symbol_expr(symbol), lhs);
    expr2tc new_lhs = lhs;

    // Again, specifiy which l1 data object we're going to make the assignment
    // to.
    renaming::level2t::rename_to_record(new_lhs, *it);

    cur_state->assignment(new_lhs, rhs, true);

    target->assignment(
      gen_true_expr(),
      new_lhs, lhs,
      rhs,
      cur_state->source,
      cur_state->gen_stack_trace(),
      symex_targett::HIDDEN);
  }
}

void
goto_symext::loop_bound_exceeded(const expr2tc &guard)
{
  const irep_idt &loop_id = cur_state->source.pc->location.loopid();

  expr2tc negated_cond;

  if (is_true(guard)) {
    negated_cond = gen_false_expr();
  } else {
    negated_cond = not2tc(guard);
  }

  if (base_case || inductive_step)
  {
    // generate unwinding assumption
    expr2tc guarded_expr=negated_cond;
    cur_state->guard.guard_expr(guarded_expr);
    target->assumption(cur_state->guard.as_expr(), guarded_expr, cur_state->source);

    // add to state guard to prevent further assignments
    cur_state->guard.add(negated_cond);
  }
  else if (forward_condition)
  {
    // generate unwinding assertion
    claim(negated_cond,
      "unwinding assertion loop "+id2string(loop_id));

    // add to state guard to prevent further assignments
    cur_state->guard.add(negated_cond);
  }
  else if(!partial_loops)
  {
    if(!no_unwinding_assertions)
    {
      // generate unwinding assertion
      claim(negated_cond, "unwinding assertion loop " + id2string(loop_id));
    } else   {
      // generate unwinding assumption, unless we permit partial loops
      expr2tc guarded_expr = negated_cond;
      cur_state->guard.guard_expr(guarded_expr);
      target->assumption(cur_state->guard.as_expr(), guarded_expr,
                         cur_state->source);
    }

    // add to state guard to prevent further assignments
    cur_state->guard.add(negated_cond);
  }
}

bool
goto_symext::get_unwind(
  const symex_targett::sourcet &source, unsigned unwind)
{
  unsigned id = source.pc->loop_number;
  unsigned long this_loop_max_unwind = max_unwind;

  if (unwind_set.count(id) != 0)
    this_loop_max_unwind = unwind_set[id];

  if (!options.get_bool_option("quiet"))
  {
    std::string msg =
      "Unwinding loop " + i2string(id) + " iteration " + i2string(unwind) +
      " " + source.pc->location.as_string();
    std::cout << msg << std::endl;
  }

  return this_loop_max_unwind != 0 &&
         unwind >= this_loop_max_unwind;
}

hash_set_cont<irep_idt, irep_id_hash> goto_symext::body_warnings;
