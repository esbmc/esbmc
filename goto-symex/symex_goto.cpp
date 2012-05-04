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
goto_symext::symex_goto(const exprt &old_guard)
{
  const goto_programt::instructiont &instruction = *cur_state->source.pc;

  exprt new_guard = old_guard;
  cur_state->rename(new_guard);
  do_simplify(new_guard);

  if (new_guard.is_false() ||
      cur_state->guard.is_false()) {
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

    if (new_guard.is_true()) {
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
  if (new_guard.is_true()) {
    cur_state->guard.make_false();
  } else   {
    // produce new guard symbol
    exprt guard_expr;

    if (new_guard.id() == exprt::symbol ||
        (new_guard.id() == exprt::i_not &&
         new_guard.operands().size() == 1 &&
         new_guard.op0().id() == exprt::symbol))
      guard_expr = new_guard;
    else {
      guard_expr = symbol_exprt(guard_identifier(), bool_typet());
      exprt new_rhs = new_guard,
            rhs = old_guard;
      new_rhs.make_not();
      rhs.make_not();

      exprt new_lhs = guard_expr;

      cur_state->assignment(new_lhs, new_rhs, false);

      guardt guard;

      expr2tc guard2, new_lhs2, guard_expr2, new_rhs2;
      migrate_expr(guard.as_expr(), guard2);
      migrate_expr(new_lhs, new_lhs2);
      migrate_expr(new_rhs, new_rhs2);
      migrate_expr(guard_expr, guard_expr2);
      target->assignment(
        guard2,
        new_lhs2, guard_expr2,
        new_rhs2,
        cur_state->source,
        cur_state->gen_stack_trace(),
        symex_targett::HIDDEN);

      guard_expr.make_not();
      cur_state->rename(guard_expr);
    }

    if (forward) {
      new_state.guard.add(guard_expr);
      guard_expr.make_not();
      cur_state->guard.add(guard_expr);
    } else   {
      cur_state->guard.add(guard_expr);
      guard_expr.make_not();
      new_state.guard.add(guard_expr);
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

      typet type(symbol.type);

      exprt rhs;

      if (cur_state->guard.is_false()) {
	rhs = symbol_exprt(cur_state->current_name(goto_state, symbol.name), type);
      } else if (goto_state.guard.is_false())    {
	rhs = symbol_exprt(cur_state->current_name(symbol.name), type);
      } else   {
	guardt tmp_guard(goto_state.guard);

	// this gets the diff between the guards
	tmp_guard -= cur_state->guard;

	rhs = if_exprt();
	rhs.type() = type;
	rhs.op0() = tmp_guard.as_expr();
	rhs.op1() = symbol_exprt(cur_state->current_name(goto_state, symbol.name),
                                 type);
	rhs.op2() = symbol_exprt(cur_state->current_name(symbol.name), type);
      }

      exprt lhs(symbol_expr(symbol));
      exprt new_lhs(lhs);

      cur_state->assignment(new_lhs, rhs, false);

      guardt true_guard;

      expr2tc true_guard2, new_lhs2, lhs2, rhs2;
      migrate_expr(true_guard.as_expr(), true_guard2);
      migrate_expr(new_lhs, new_lhs2);
      migrate_expr(lhs, lhs2);
      migrate_expr(rhs, rhs2);
      target->assignment(
        true_guard2,
        new_lhs2, lhs2,
        rhs2,
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
goto_symext::loop_bound_exceeded(const exprt &guard)
{
  const irep_idt &loop_id = cur_state->source.pc->location.loopid();

  exprt negated_cond;

  if (guard.is_true())
    negated_cond = false_exprt();
  else
    negated_cond = gen_not(guard);

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
      exprt guarded_expr = negated_cond;
      cur_state->guard.guard_expr(guarded_expr);
      target->assumption(cur_state->guard, guarded_expr, cur_state->source);
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
