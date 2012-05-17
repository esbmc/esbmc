/*******************************************************************\

   Module: Symbolic Execution

   Author: Daniel Kroening, kroening@kroening.com Lucas Cordeiro,
     lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>
#include <assert.h>

#include <expr_util.h>
#include <std_expr.h>

#include "goto_symex.h"

void
goto_symext::symex_goto(const expr2tc &old_guard)
{
  const goto_programt::instructiont &instruction = *cur_state->source.pc;

  expr2tc new_guard = old_guard;
  cur_state->rename(new_guard);
  do_simplify(new_guard);

  if ((is_constant_bool2t(new_guard) &&
        !to_constant_bool2t(new_guard).constant_value)
      || cur_state->guard.is_false()) {

    // reset unwinding counter
    cur_state->unwind_map[cur_state->source] = 0;

    // next instruction
    cur_state->source.pc++;

    return; // nothing to do
  }

  assert(!instruction.targets.empty());

  // we only do deterministic gotos for now
  if (instruction.targets.size() != 1)
    throw "no support for non-deterministic gotos";

  goto_programt::const_targett goto_target =
    instruction.targets.front();

  bool forward =
    cur_state->source.pc->location_number <
    goto_target->location_number;

  if (!forward) { // backwards?
    unsigned unwind;

    unwind = cur_state->unwind_map[cur_state->source];
    unwind++;
    cur_state->unwind_map[cur_state->source] = unwind;

    if (get_unwind(cur_state->source, unwind)) {
      loop_bound_exceeded(new_guard);

      // reset unwinding
      cur_state->unwind_map[cur_state->source] = 0;

      // next instruction
      cur_state->source.pc++;
      return;
    }

    if (is_constant_bool2t(new_guard) &&
        to_constant_bool2t(new_guard).constant_value) {
      cur_state->source.pc = goto_target;
      return; // nothing else to do
    }
  }

  goto_programt::const_targett new_state_pc, state_pc;

  if (forward) {
    new_state_pc = goto_target; // goto target instruction
    state_pc = cur_state->source.pc;
    state_pc++; // next instruction
  } else   {
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
  if (is_constant_bool2t(new_guard) &&
      to_constant_bool2t(new_guard).constant_value) {
    cur_state->guard.make_false();
  } else   {
    // produce new guard symbol
    expr2tc guard_expr;

    if (is_symbol2t(new_guard) ||
        (is_not2t(new_guard) && is_symbol2t(to_not2t(new_guard).value))) {
      guard_expr = new_guard;
    } else {
      guard_expr =
        expr2tc(new symbol2t(type_pool.get_bool(), guard_identifier()));

      expr2tc new_rhs = new_guard;
      new_rhs = expr2tc(new not2t(new_rhs));
      do_simplify(new_rhs);

      expr2tc new_lhs = guard_expr;

      cur_state->assignment(new_lhs, new_rhs, false);

      guardt guard;

      expr2tc guard2;
      migrate_expr(guard.as_expr(), guard2);
      target->assignment(
        guard2,
        new_lhs, guard_expr,
        new_rhs,
        cur_state->source,
        cur_state->gen_stack_trace(),
        symex_targett::HIDDEN);

      guard_expr = expr2tc(new not2t(guard_expr));
      cur_state->rename(guard_expr);
    }

    exprt tmp_guard_expr = migrate_expr_back(guard_expr);
    if (forward) {
      new_state.guard.add(tmp_guard_expr);
      tmp_guard_expr.make_not();
      cur_state->guard.add(tmp_guard_expr);
    } else   {
      cur_state->guard.add(tmp_guard_expr);
      tmp_guard_expr.make_not();
      new_state.guard.add(tmp_guard_expr);
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
  std::set<irep_idt> variables;

  goto_state.level2.get_variables(variables);
  cur_state->level2.get_variables(variables);

  for (std::set<irep_idt>::const_iterator
       it = variables.begin();
       it != variables.end();
       it++)
  {
    if (goto_state.level2.current_number(*it) ==
        cur_state->level2.current_number(*it))
      continue;  // not changed

    if (*it == guard_identifier())
      continue;  // just a guard

    irep_idt original_identifier = cur_state->get_original_name(*it);
    try
    {
      // changed!
      const symbolt &symbol = ns.lookup(original_identifier);

      type2tc type;
      typet old_type = symbol.type;
      migrate_type(symbol.type, type);

      exprt tmp_rhs;
      expr2tc rhs;

      if (cur_state->guard.is_false()) {
	tmp_rhs = symbol_exprt(cur_state->current_name(goto_state, symbol.name),
                               old_type);
        migrate_expr(tmp_rhs, rhs);
      } else if (goto_state.guard.is_false())    {
	tmp_rhs = symbol_exprt(cur_state->current_name(symbol.name), old_type);
        migrate_expr(tmp_rhs, rhs);
      } else   {
	guardt tmp_guard(goto_state.guard);

	// this gets the diff between the guards
	tmp_guard -= cur_state->guard;

        expr2tc cond;
        migrate_expr(tmp_guard.as_expr(), cond);
	expr2tc true_val =
          expr2tc(new symbol2t(type,
                             cur_state->current_name(goto_state, symbol.name)));
        expr2tc false_val = expr2tc(new symbol2t(type,
                                         cur_state->current_name(symbol.name)));
        rhs = expr2tc(new if2t(type, cond, true_val, false_val));
      }

      exprt tmp_lhs(symbol_expr(symbol));
      expr2tc lhs;
      migrate_expr(tmp_lhs, lhs);
      expr2tc new_lhs = lhs;

      cur_state->assignment(new_lhs, rhs, false);

      guardt true_guard;

      expr2tc true_guard2;
      migrate_expr(true_guard.as_expr(), true_guard2);
      target->assignment(
        true_guard2,
        new_lhs, lhs,
        rhs,
        cur_state->source,
        cur_state->gen_stack_trace(),
        symex_targett::HIDDEN);
    }
    catch (const std::string e)
    {
      continue;
    }
  }
}

void
goto_symext::loop_bound_exceeded(const expr2tc &guard)
{
  const irep_idt &loop_id = cur_state->source.pc->location.loopid();

  expr2tc negated_cond;

  if (is_constant_bool2t(guard) && to_constant_bool2t(guard).constant_value) {
    negated_cond = expr2tc(new constant_bool2t(false));
  } else {
    negated_cond = expr2tc(new not2t(guard));
  }

  bool unwinding_assertions =
    !options.get_bool_option("no-unwinding-assertions");

  bool partial_loops =
    options.get_bool_option("partial-loops");

  if (!partial_loops) {
    if (unwinding_assertions) {
      // generate unwinding assertion
      claim(negated_cond, "unwinding assertion loop " + id2string(loop_id));
    } else   {
      // generate unwinding assumption, unless we permit partial loops
      exprt tmp_negated_cond = migrate_expr_back(negated_cond);
      cur_state->guard.guard_expr(tmp_negated_cond);
      migrate_expr(tmp_negated_cond, negated_cond);

      expr2tc guard, guarded;
      migrate_expr(cur_state->guard.as_expr(), guard);
      guarded = negated_cond;
      target->assumption(guard, guarded, cur_state->source);
    }

    // add to state guard to prevent further assignments
    exprt tmp_negated_cond = migrate_expr_back(negated_cond);
    cur_state->guard.add(tmp_negated_cond);
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

  #if 1
  {
    std::string msg =
      "Unwinding loop " + i2string(id) + " iteration " + i2string(unwind) +
      " " + source.pc->location.as_string();
    std::cout << msg << std::endl;
  }
  #endif

  return this_loop_max_unwind != 0 &&
         unwind >= this_loop_max_unwind;
}
