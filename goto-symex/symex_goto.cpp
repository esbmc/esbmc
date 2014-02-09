/*******************************************************************\

   Module: Symbolic Execution

   Author: Daniel Kroening, kroening@kroening.com Lucas Cordeiro,
     lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>
#include <assert.h>
#include <prefix.h>

#include <expr_util.h>
#include <std_expr.h>

#include "goto_symex.h"
#include "symex_target_equation.h"

void
goto_symext::enter_insn()
{
  merge_gotos();

  // Store the current set of loops we're in.
  cur_state->top().cur_loops = cur_state->source.pc->loop_membership;
  if (cur_state->source.pc->type == FUNCTION_CALL ||
      cur_state->source.pc->type == END_FUNCTION)
    cur_state->check_loop_structure = false;
  else if (!cur_state->source.pc->function->loops_well_formed)
    cur_state->check_loop_structure = false;
  else
    cur_state->check_loop_structure = true;
}

void
goto_symext::exit_insn()
{

  if (!cur_state->check_loop_structure)
    return;

  check_loop_transitions(cur_state->top().cur_loops,
      cur_state->source.pc->loop_membership, cur_state->guard);
}

void
goto_symext::check_loop_transitions(const loop_membershipt &old,
    const loop_membershipt &now, const guardt &newguard)
{

  // Check whether or not loop status has changed.
  loop_transitionst loop_changes = find_loop_transitions(old, now);

  // Only thing at the insn level that we care about at this stage is the
  // initial entry to the loop, not via any merged states.
  for (const auto &thepair : loop_changes) {
    if (thepair.second) {
      cur_state->top().loop_entry_guards[thepair.first].push_back(newguard);
    } else {
      // On exit from a loop for whatever reason, clear remaining accounting
      // data.
      cur_state->top().loop_entry_guards[thepair.first].clear();
      cur_state->top().loop_exit_guards[thepair.first].clear();
      cur_state->top().loop_assumpts[thepair.first].clear();
      cur_state->top().prev_loop_guards[thepair.first].make_true();
    }
  }
}

goto_symext::loop_transitionst
goto_symext::find_loop_transitions(
   const goto_programt::instructiont::loop_membershipt &old,
   const goto_programt::instructiont::loop_membershipt &now)
{
  std::vector<std::pair<unsigned int, bool> > returnvec;

  // OK, that's changed. Now, have we entered or exited a loop?
  for (unsigned int loopid : old) {
    if (now.find(loopid) == now.end()) {
      // We've left a loop.
      returnvec.push_back(std::make_pair(loopid, false));
    }
  }

  // How about entry?
  for (unsigned int loopid : now) {
    if (old.find(loopid) == old.end()) {
      // Entered a loop.
      returnvec.push_back(std::make_pair(loopid, true));
    }
  }

  return returnvec;
}

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

  if (!new_guard_false && options.get_bool_option("smt-symex-guard")) {
    runtime_encoded_equationt *rte = dynamic_cast<runtime_encoded_equationt*>
                                                 (target);
    equality2tc question(true_expr, new_guard);
    try {
      tvt res = rte->ask_solver_question(question);

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

  if (new_guard_false) {

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

    if (new_guard_true) {
      fix_backwards_goto_guard(cur_state->source.pc->loop_number, true_expr);
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

  symex_targett::sourcet src_insn = cur_state->source;
  cur_state->source.pc = state_pc;

  // put into state-queue
  statet::goto_state_listt &goto_state_list =
    cur_state->top().goto_state_map[new_state_pc];

  goto_state_list.push_back(statet::goto_statet(*cur_state));
  statet::goto_statet &new_state = goto_state_list.back();

  // adjust guards
  expr2tc guard_expr;
  if (new_guard_true) {
    cur_state->guard.make_false();
    guard_expr = true_expr;
  } else {
    // produce new guard symbol

    if (is_symbol2t(new_guard) ||
        (is_not2t(new_guard) && is_symbol2t(to_not2t(new_guard).value))) {
      guard_expr = new_guard;
    } else {
      guard_expr = guard_identifier();

      expr2tc new_rhs = new_guard;
      new_rhs = not2tc(new_rhs);
      do_simplify(new_rhs);

      cur_state->assignment(guard_expr, new_rhs, false);

      guardt guard;

      target->assignment(
        guard.as_expr(),
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

  // Record exits from loops. Ordering (backwards-false / forwards-true)
  // already handled via indirection above.
  loop_transitionst exits =
    find_loop_transitions(src_insn.pc->loop_membership,
                          new_state_pc->loop_membership);
  for (auto &item : exits) {
    if (!item.second) {
      cur_state->top().loop_exit_guards[item.first].push_back(new_state.guard);
    }
  }

  if (!forward)
    fix_backwards_goto_guard(src_insn.pc->loop_number, guard_expr);
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

    // If we've merged into a loop, record as an entry. Ignore exits.
    loop_transitionst transitions =
      find_loop_transitions(goto_state.src_loops,
                            cur_state->source.pc->loop_membership);
    for (auto &i : transitions) {
      if (i.second) {
        cur_state->top().loop_entry_guards[i.first].push_back(goto_state.guard);
      }
    }

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

    if (cur_state->guard.is_false()) {
      rhs = symbol2tc(type, symbol.name);
      cur_state->current_name(goto_state, rhs);
    } else if (goto_state.guard.is_false())    {
      rhs = symbol2tc(type, symbol.name);
      cur_state->current_name(goto_state, rhs);
    } else   {
      guardt tmp_guard(goto_state.guard);

      // this gets the diff between the guards
      tmp_guard -= cur_state->guard;

      symbol2tc true_val(type, symbol.name);
      symbol2tc false_val(type, symbol.name);
      cur_state->current_name(goto_state, true_val);
      cur_state->current_name(false_val);
      rhs = if2tc(type, tmp_guard.as_expr(), true_val, false_val);
    }

    exprt tmp_lhs(symbol_expr(symbol));
    expr2tc lhs;
    migrate_expr(tmp_lhs, lhs);
    expr2tc new_lhs = lhs;

    cur_state->assignment(new_lhs, rhs, false);

    guardt true_guard;

    target->assignment(
      true_guard.as_expr(),
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
    negated_cond = false_expr;
  } else {
    negated_cond = not2tc(guard);
  }

  if (base_case)
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

expr2tc
goto_symext::assign_guard_symbol(std::string basename, const expr2tc &val)
{
  irep_idt name(basename);
  const symbolt *sym = NULL;
  if (ns.lookup(name, sym)) {
    symbolt newsym;
    newsym.name = name;
    newsym.type = bool_typet();
    newsym.base_name = name;
    new_context.add(newsym);
    sym = &newsym;
  }

  exprt tmp_lhs(symbol_expr(*sym));
  expr2tc lhs;
  migrate_expr(tmp_lhs, lhs);
  expr2tc new_lhs = lhs;

  cur_state->assignment(new_lhs, val, false);

  guardt true_guard;

  target->assignment(
    true_guard.as_expr(),
    new_lhs, lhs,
    val,
    cur_state->source,
    cur_state->gen_stack_trace(),
    symex_targett::HIDDEN);

  return new_lhs;
}


expr2tc
goto_symext::accuml_guard_symbol(std::string name,
    const std::vector<guardt> &guards,
    const guardt &prefix)
{
  guardt entry_guard;

  for (auto g : guards) {
    g -= prefix;
    entry_guard |= g;
  }

  return assign_guard_symbol(name, entry_guard.as_expr());
}

void
goto_symext::fix_backwards_goto_guard(unsigned int loopno,
    const expr2tc &continue_cond)
{
  if (!cur_state->source.pc->function->loops_well_formed)
    return;

  const guardt &old_guard = cur_state->top().prev_loop_guards[loopno];
  if (old_guard.empty()) {
    // Haven't been around this loop before. Take the entry conditions and
    // assign that to a new symbol. Same for the exit conditions. Conjoin with
    // assumpts and continuation conditions, to make new guard. Worry about
    // reducing duplicate symbols in the future.

    // Has to have been at least one entry condition.
    assert(!cur_state->top().loop_entry_guards[loopno].empty() &&
           "Every loop has to have at least one entry");

    std::stringstream ss, ss2;
    ss << "symex::entry_conds_loop_" << cur_state->source.pc->loop_number;
    expr2tc entry_sym = accuml_guard_symbol(ss.str(),
        cur_state->top().loop_entry_guards[loopno]);
    cur_state->top().loop_entry_guards[loopno].clear();

    // Optionally collect together all of the loop exit conditions.
    expr2tc exit_cond = false_expr;
    if (!cur_state->top().loop_exit_guards[loopno].empty()) {
      ss2 << "symex::exit_conds_loop_" << cur_state->source.pc->loop_number;
      exit_cond = accuml_guard_symbol(ss2.str(),
          cur_state->top().loop_exit_guards[loopno]);
      cur_state->top().loop_exit_guards[loopno].clear();
    }

    // Accumulate any assumptions into the continue condition.
    expr2tc to_continue = continue_cond;
    if (!cur_state->top().loop_assumpts[loopno].empty()) {
      std::stringstream ss3;
      ss3 << "symex::continue_conds_loop_" << cur_state->source.pc->loop_number;
      expr2tc tmp = accuml_guard_symbol(ss3.str(),
          cur_state->top().loop_assumpts[loopno]);
      cur_state->top().loop_assumpts[loopno].clear();

      to_continue = and2tc(to_continue, tmp);
    }

    // OK. Final new guard is: entry & !exit & continue
    cur_state->guard.make_true();
    cur_state->guard.add(entry_sym);
    cur_state->guard.add(not2tc(exit_cond));
    cur_state->guard.add(to_continue);

    cur_state->top().prev_loop_guards[loopno] = cur_state->guard;
  } else {
    // We _have_ been around this loop before. Thus, we shouldn't have received
    // any new entry conditions.
    assert(cur_state->top().loop_entry_guards[loopno].empty());
    const guardt &prev_guard = cur_state->top().prev_loop_guards[loopno];

    // There may have been exits though.
    expr2tc exit_cond = false_expr;
    if (!cur_state->top().loop_exit_guards[loopno].empty()) {
      std::stringstream ss2;
      ss2 << "symex::exit_conds_loop_" << cur_state->source.pc->loop_number;
      exit_cond = accuml_guard_symbol(ss2.str(),
          cur_state->top().loop_exit_guards[loopno], prev_guard);
      cur_state->top().loop_exit_guards[loopno].clear();
    }

    // Accumulate any assumptions into the continue condition.
    expr2tc to_continue = continue_cond;
    if (!cur_state->top().loop_assumpts[loopno].empty()) {
      std::stringstream ss3;
      ss3 << "symex::continue_conds_loop_" << cur_state->source.pc->loop_number;
      expr2tc tmp = accuml_guard_symbol(ss3.str(),
          cur_state->top().loop_assumpts[loopno], prev_guard);
      cur_state->top().loop_assumpts[loopno].clear();

      to_continue = and2tc(to_continue, tmp);
    }

    // OK. Revert to the guard for the previous iteration.
    cur_state->guard = prev_guard;
    // Add that we haven't exited
    cur_state->guard.add(not2tc(exit_cond));
    // And that we continue
    cur_state->guard.add(to_continue);

    cur_state->top().prev_loop_guards[loopno] = cur_state->guard;
  }
}
