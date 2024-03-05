#include <cassert>
#include <fstream>
#include <goto-symex/goto_symex.h>
#include <goto-symex/slice.h>
#include <goto-symex/symex_target_equation.h>

#include <langapi/language_ui.h>
#include <solvers/smtlib/smtlib_conv.h>
#include <util/expr_util.h>
#include <irep2/irep2.h>
#include <util/migrate.h>
#include <util/prefix.h>
#include <util/std_expr.h>

void goto_symext::symex_goto(const expr2tc &old_guard)
{
  const goto_programt::instructiont &instruction = *cur_state->source.pc;

  expr2tc new_guard = old_guard;
  cur_state->rename(new_guard);
  do_simplify(new_guard);

  bool new_guard_false = (is_false(new_guard) || cur_state->guard.is_false());
  bool new_guard_true = is_true(new_guard);

  if (!new_guard_false && options.get_bool_option("smt-symex-guard"))
  {
    auto rte = std::dynamic_pointer_cast<runtime_encoded_equationt>(target);

    expr2tc question = equality2tc(gen_true_expr(), new_guard);
    try
    {
      tvt res = rte->ask_solver_question(question);

      if (res.is_false())
        new_guard_false = true;
      else if (res.is_true())
        new_guard_true = true;
    }
    catch (runtime_encoded_equationt::dual_unsat_exception &e)
    {
      // Assumptions mean that the guard is never satisfiable as true or false,
      // basically means we've assume'd away the possibility of hitting this
      // point.
      new_guard_false = true;
    }
  }

  goto_programt::const_targett goto_target = instruction.targets.front();

  bool forward =
    cur_state->source.pc->location_number < goto_target->location_number;

  if (new_guard_false)
  {
    // reset unwinding counter
    if (instruction.is_backwards_goto())
    {
      cur_state->loop_iterations[instruction.loop_number] = 0;

      // Reset loop counter
      if (instruction.loop_number == first_loop)
        first_loop = 0;
    }

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
    if (goto_target == cur_state->source.pc)
    {
      assert(
        cur_state->source.pc->location_number == goto_target->location_number);

      // generate assume(false) or a suitable negation if this
      // instruction is a conditional goto
      if (new_guard_true)
        assume(gen_false_expr());
      else
      {
        make_not(new_guard);
        assume(new_guard);
      }

      // next instruction
      cur_state->source.pc++;
      return;
    }

    BigInt &unwind = cur_state->loop_iterations[instruction.loop_number];
    ++unwind;

    if (get_unwind(cur_state->source, unwind))
    {
      loop_bound_exceeded(new_guard);

      // reset unwinding
      unwind = 0;

      // next instruction
      cur_state->source.pc++;

      // Reset loop counter
      if (instruction.loop_number == first_loop)
        first_loop = 0;

      return;
    }

    if (new_guard_true)
    {
      cur_state->source.pc = goto_target;
      return; // nothing else to do
    }
  }

  goto_programt::const_targett new_state_pc, state_pc;

  if (forward)
  {
    new_state_pc = goto_target; // goto target instruction
    state_pc = cur_state->source.pc;
    state_pc++; // next instruction
  }
  else
  {
    new_state_pc = cur_state->source.pc;
    new_state_pc++;
    state_pc = goto_target;
  }

  cur_state->source.pc = state_pc;

  // put into state-queue
  statet::goto_state_listt &goto_state_list =
    cur_state->top().goto_state_map[new_state_pc];

  goto_state_list.emplace_back(*cur_state);

  // adjust guards
  if (new_guard_true)
  {
    cur_state->guard.make_false();
  }
  else
  {
    statet::goto_statet &new_state = goto_state_list.back();

    // produce new guard symbol
    expr2tc guard_expr;

    if (
      is_symbol2t(new_guard) ||
      (is_not2t(new_guard) && is_symbol2t(to_not2t(new_guard).value)))
    {
      guard_expr = new_guard;
    }
    else
    {
      guard_expr = guard_identifier();

      expr2tc new_rhs = new_guard;
      new_rhs = not2tc(new_rhs);
      do_simplify(new_rhs);

      cur_state->assignment(guard_expr, new_rhs);

      target->assignment(
        gen_true_expr(),
        guard_expr,
        guard_expr,
        new_rhs,
        expr2tc(),
        cur_state->source,
        cur_state->gen_stack_trace(),
        true,
        first_loop);

      if (is_constant_expr(new_rhs))
        guard_expr = new_rhs;

      guard_expr = not2tc(guard_expr);
      do_simplify(guard_expr);
    }

    expr2tc not_guard_expr = not2tc(guard_expr);
    do_simplify(not_guard_expr);

    if (forward)
    {
      new_state.guard.add(guard_expr);
      cur_state->guard.add(not_guard_expr);
    }
    else
    {
      cur_state->guard.add(guard_expr);
      new_state.guard.add(not_guard_expr);
    }
  }
}

static inline guardt merge_state_guards(
  goto_symext::statet::goto_statet &goto_state,
  goto_symex_statet &state)
{
  // adjust guard, even using guards from unreachable states. This helps to
  // shrink the state guard if the incoming edge is from a path that was
  // truncated by config.unwind, config.depth or an assume-false instruction.

  // Note when an unreachable state contributes its guard, merging it in is
  // optional, since the formula already implies the unreachable guard is
  // impossible. Therefore we only integrate it when to do so simplifies the
  // state guard.

  // In CBMC this function can trash either state's guards, since goto_state is
  // dying and state's guard will shortly be overwritten. However, we still use
  // either state's guard, so keep them intact.
  if (
    (!goto_state.guard.is_false() && !state.guard.is_false()) ||
    state.guard.disjunction_may_simplify(goto_state.guard))
  {
    state.guard |= goto_state.guard;
    return state.guard;
  }
  else if (state.guard.is_false() && !goto_state.guard.is_false())
  {
    return goto_state.guard;
  }
  else
  {
    return state.guard;
  }
}

void goto_symext::merge_gotos()
{
  statet::framet &frame = cur_state->top();

  // first, see if this is a target at all
  statet::goto_state_mapt::iterator state_map_it =
    frame.goto_state_map.find(cur_state->source.pc);

  if (state_map_it == frame.goto_state_map.end())
    return; // nothing to do

  // we need to merge
  statet::goto_state_listt &state_list = state_map_it->second;

  for (auto list_it = state_list.rbegin(); list_it != state_list.rend();
       list_it++)
  {
    statet::goto_statet &goto_state = *list_it;

    // Merge guards. Don't write this to `state` yet because we might move
    // goto_state over it below.
    guardt new_guard = merge_state_guards(goto_state, *cur_state);

    if (!goto_state.guard.is_false())
    {
      // do SSA phi functions
      phi_function(goto_state);

      merge_locality(goto_state);

      merge_value_sets(goto_state);

      // adjust depth
      cur_state->num_instructions =
        std::min(cur_state->num_instructions, goto_state.num_instructions);
    }

    cur_state->guard = std::move(new_guard);
  }

  // clean up to save some memory
  frame.goto_state_map.erase(state_map_it);
}

void goto_symext::merge_locality(const statet::goto_statet &src)
{
  if (cur_state->guard.is_false())
  {
    cur_state->top().local_variables = src.local_variables;
    return;
  }

  cur_state->top().local_variables.insert(
    src.local_variables.begin(), src.local_variables.end());
}

void goto_symext::merge_value_sets(const statet::goto_statet &src)
{
  if (cur_state->guard.is_false())
  {
    cur_state->value_set = src.value_set;
    return;
  }

  cur_state->value_set.make_union(src.value_set, true);
}

void goto_symext::phi_function(const statet::goto_statet &goto_state)
{
  if (goto_state.guard.is_false() && cur_state->guard.is_false())
    return;

  // go over all variables to see what changed
  const auto &variables = cur_state->level2.current_names;

  const auto &goto_variables = goto_state.level2.current_names;

  guardt tmp_guard;
  if (
    !variables.empty() && !cur_state->guard.is_false() &&
    !goto_state.guard.is_false())
  {
    tmp_guard = goto_state.guard;

    // this gets the diff between the guards
    tmp_guard -= cur_state->guard;
  }

  for (const auto &[variable, value] : variables)
  {
    if (variable.base_name == guard_identifier_s)
      continue; // just a guard

    if (has_prefix(variable.base_name, "symex::invalid_object"))
      continue;

    auto goto_value = goto_variables.find(variable);
    // If the variable was deleted in this branch, don't create an assignment
    // for it
    if (goto_value == goto_variables.end())
      continue;

    if (goto_value->second.count == value.count)
      continue; // not changed

    // changed!
    const symbolt &symbol = *ns.lookup(variable.base_name);

    type2tc type = migrate_type(symbol.type);

    expr2tc cur_state_rhs = symbol2tc(type, symbol.id);
    renaming::level2t::rename_to_record(cur_state_rhs, variable);

    expr2tc goto_state_rhs = symbol2tc(type, symbol.id);
    renaming::level2t::rename_to_record(goto_state_rhs, variable);

    expr2tc rhs;
    // Semi-manually rename these symbols: we may be referring to an l1
    // variable not in the current scope, thus we need to directly specify
    // which l1 variable we're dealing with.
    goto_state.level2.rename(goto_state_rhs);
    if (cur_state->guard.is_false())
      rhs = goto_state_rhs;

    cur_state->level2.rename(cur_state_rhs);
    if (goto_state.guard.is_false())
      rhs = cur_state_rhs;
    else
    {
      rhs = if2tc(type, tmp_guard.as_expr(), goto_state_rhs, cur_state_rhs);
      simplify(rhs);
    }

    expr2tc lhs;
    migrate_expr(symbol_expr(symbol), lhs);
    expr2tc new_lhs = lhs;

    // Again, specifiy which l1 data object we're going to make the assignment
    // to.
    renaming::level2t::rename_to_record(new_lhs, variable);

    cur_state->rename_type(new_lhs);
    cur_state->rename_type(rhs);
    cur_state->assignment(new_lhs, rhs);

    target->assignment(
      gen_true_expr(),
      new_lhs,
      lhs,
      rhs,
      expr2tc(),
      cur_state->source,
      cur_state->gen_stack_trace(),
      true,
      first_loop);
  }
}

void goto_symext::loop_bound_exceeded(const expr2tc &guard)
{
  if (partial_loops && !config.options.get_bool_option("termination"))
    return;

  unsigned loop_number = cur_state->source.pc->loop_number;

  expr2tc negated_cond = guard;
  make_not(negated_cond);

  if (!no_unwinding_assertions)
  {
    // generate unwinding assertion
    claim(negated_cond, "unwinding assertion loop " + i2string(loop_number));
  }
  else
  {
    // generate unwinding assumption, unless we permit partial loops
    expr2tc guarded_expr = negated_cond;
    cur_state->guard.guard_expr(guarded_expr);
    target->assumption(
      cur_state->guard.as_expr(), guarded_expr, cur_state->source, first_loop);
  }

  // add to state guard to prevent further assignments
  cur_state->guard.add(negated_cond);
}

bool goto_symext::get_unwind(
  const symex_targett::sourcet &source,
  const BigInt &unwind)
{
  unsigned id = source.pc->loop_number;
  BigInt this_loop_max_unwind = max_unwind;

  if (unwind_set.count(id) != 0)
    this_loop_max_unwind = unwind_set[id];

  bool stop_unwind =
    this_loop_max_unwind != 0 && unwind >= this_loop_max_unwind;
  if (!options.get_bool_option("quiet"))
  {
    log_status(
      "{} loop {} iteration {}   {}",
      stop_unwind ? "Not unwinding" : "Unwinding",
      i2string(cur_state->source.pc->loop_number),
      integer2string(unwind),
      cur_state->source.pc->location);
  }

  return stop_unwind;
}
