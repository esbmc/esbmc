/*******************************************************************\

   Module:

   Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <boost/shared_ptr.hpp>
#include <goto-symex/execution_state.h>
#include <goto-symex/reachability_tree.h>
#include <langapi/language_ui.h>
#include <langapi/languages.h>
#include <langapi/mode.h>
#include <sstream>
#include <string>
#include <util/c_types.h>
#include <util/config.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/irep2.h>
#include <util/migrate.h>
#include <util/simplify_expr.h>
#include <util/std_expr.h>
#include <util/string2array.h>
#include <vector>

unsigned int execution_statet::node_count = 0;
unsigned int execution_statet::dynamic_counter = 0;
std::map<expr2tc, std::list<unsigned int>> vars_map;
std::map<expr2tc, bool> is_global;

execution_statet::execution_statet(
  const goto_functionst &goto_functions,
  const namespacet &ns,
  reachability_treet *art,
  boost::shared_ptr<symex_targett> _target,
  contextt &context,
  boost::shared_ptr<ex_state_level2t> l2init,
  optionst &options,
  message_handlert &_message_handler)
  : goto_symext(ns, context, goto_functions, std::move(_target), options),
    owning_rt(art),
    state_level2(std::move(l2init)),
    global_value_set(ns),
    message_handler(_message_handler)
{
  art1 = owning_rt;
  CS_number = 0;
  TS_number = 0;
  node_id = 0;
  tid_is_set = false;
  monitor_tid = 0;
  mon_from_tid = false;
  monitor_from_tid = 0;
  guard_execution = "execution_statet::\\guard_exec";
  interleaving_unviable = false;
  symex_trace = options.get_bool_option("symex-trace");
  smt_during_symex = options.get_bool_option("smt-during-symex");
  smt_thread_guard = options.get_bool_option("smt-thread-guard");

  goto_functionst::function_mapt::const_iterator it =
    goto_functions.function_map.find("__ESBMC_main");
  if(it == goto_functions.function_map.end())
  {
    std::cerr << "main symbol not found; please set an entry point"
              << std::endl;
    abort();
  }

  const goto_programt *goto_program = &(it->second.body);

  // Initialize initial thread state
  goto_symex_statet state(*state_level2, global_value_set, ns);
  state.initialize(
    (*goto_program).instructions.begin(),
    (*goto_program).instructions.end(),
    goto_program,
    0);

  threads_state.push_back(state);
  preserved_paths.emplace_back();
  cur_state = &threads_state.front();
  cur_state->global_guard.make_true();
  cur_state->global_guard.add(get_guard_identifier());

  atomic_numbers.push_back(0);

  if(DFS_traversed.size() <= state.source.thread_nr)
  {
    DFS_traversed.push_back(false);
  }
  else
  {
    DFS_traversed[state.source.thread_nr] = false;
  }

  thread_start_data.emplace_back();

  // Initial mpor tracking.
  thread_last_reads.emplace_back();
  thread_last_writes.emplace_back();
  // One thread with one dependancy relation.
  dependancy_chain.emplace_back();
  dependancy_chain.back().push_back(0);
  mpor_says_no = false;

  cswitch_forced = false;
  active_thread = 0;
  last_active_thread = 0;
  last_insn = nullptr;
  node_count = 0;
  nondet_count = 0;
  DFS_traversed.reserve(1);
  DFS_traversed[0] = false;
  mon_thread_warning = false;

  thread_cswitch_threshold = 2;
}

execution_statet::execution_statet(const execution_statet &ex)
  : goto_symext(ex),
    owning_rt(ex.owning_rt),
    state_level2(
      boost::dynamic_pointer_cast<ex_state_level2t>(ex.state_level2->clone())),
    global_value_set(ex.global_value_set),
    message_handler(ex.message_handler)
{
  *this = ex;

  // Regenerate threads state using new objects state_level2 ref
  threads_state.clear();
  std::vector<goto_symex_statet>::const_iterator it;
  for(it = ex.threads_state.begin(); it != ex.threads_state.end(); it++)
  {
    goto_symex_statet state(*it, *state_level2, global_value_set);
    threads_state.push_back(state);
  }

  // Reassign which state is currently being worked on.
  cur_state = &threads_state[active_thread];
}

execution_statet &execution_statet::operator=(const execution_statet &ex)
{
  // Don't copy level2, copy cons it in execution_statet(ref)
  //state_level2 = ex.state_level2;

  threads_state = ex.threads_state;
  preserved_paths = ex.preserved_paths;
  atomic_numbers = ex.atomic_numbers;
  DFS_traversed = ex.DFS_traversed;
  thread_start_data = ex.thread_start_data;
  last_active_thread = ex.last_active_thread;
  last_insn = ex.last_insn;
  active_thread = ex.active_thread;
  guard_execution = ex.guard_execution;
  nondet_count = ex.nondet_count;
  node_id = ex.node_id;
  global_value_set = ex.global_value_set;
  interleaving_unviable = ex.interleaving_unviable;
  pre_goto_guard = ex.pre_goto_guard;
  mon_thread_warning = ex.mon_thread_warning;

  monitor_tid = ex.monitor_tid;
  tid_is_set = ex.tid_is_set;
  monitor_from_tid = ex.monitor_from_tid;
  mon_from_tid = ex.mon_from_tid;
  thread_cswitch_threshold = ex.thread_cswitch_threshold;
  symex_trace = ex.symex_trace;
  smt_during_symex = ex.smt_during_symex;
  smt_thread_guard = ex.smt_thread_guard;
  stack_limit = ex.stack_limit;
  no_return_value_opt = ex.no_return_value_opt;

  CS_number = ex.CS_number;
  TS_number = ex.TS_number;

  thread_last_reads = ex.thread_last_reads;
  thread_last_writes = ex.thread_last_writes;
  dependancy_chain = ex.dependancy_chain;
  mpor_says_no = ex.mpor_says_no;
  cswitch_forced = ex.cswitch_forced;

  // Vastly irritatingly, we have to iterate through existing level2t objects
  // updating their ex_state references. There isn't an elegant way of updating
  // them, it seems, while keeping the symex stuff ignorant of ex_state.
  // Oooooo, so this is where auto types would be useful...
  for(auto &it : threads_state)
  {
    for(goto_symex_statet::call_stackt::iterator it2 = it.call_stack.begin();
        it2 != it.call_stack.end();
        it2++)
    {
      for(auto &it3 : it2->goto_state_map)
      {
        for(goto_symex_statet::goto_state_listt::iterator it4 =
              it3.second.begin();
            it4 != it3.second.begin();
            it4++)
        {
          ex_state_level2t &l2 = dynamic_cast<ex_state_level2t &>(it4->level2);
          l2.owner = this;
        }
      }
    }
  }

  state_level2->owner = this;

  return *this;
}

void trap_to_python(reachability_treet *art);

void execution_statet::symex_step(reachability_treet &art)
{
  statet &state = get_active_state();
  const goto_programt::instructiont &instruction = *state.source.pc;
  last_insn = &instruction;

  merge_gotos();

  if(break_insn != 0 && break_insn == instruction.location_number)
  {
// This twisty turny passage of ifdefs traps to python if esbmc has been
// built with python support; otherwise it executes a trap instruction.
#ifdef WITH_PYTHON
    std::cerr << "Trapping to python; call esbmc.trap() to break if you have "
                 "gdb attached, then walk back up to symex frames"
              << std::endl;
    trap_to_python(owning_rt);
#else
#ifndef _WIN32
    __asm__("int $3");
#else
    std::cerr << "Can't trap on windows, sorry" << std::endl;
    abort();
#endif
#endif
  }

  // Don't convert if it's a inductive instruction and we are running the base
  // case or forward condition
  if((base_case || forward_condition) && instruction.inductive_step_instruction)
  {
    cur_state->source.pc++;
    return;
  }

  if(options.get_bool_option("show-symex-value-sets"))
  {
    std::cout << '\n';
    state.value_set.dump();
  }

  if(symex_trace || options.get_bool_option("show-symex-value-sets"))
    state.source.pc->output_instruction(ns, "", std::cout, false);

  switch(instruction.type)
  {
  case END_FUNCTION:
    if(instruction.function == "__ESBMC_main")
    {
      end_thread();
      force_cswitch();
    }
    else
    {
      // Fall through to base class
      goto_symext::symex_step(art);
    }
    break;
  case ATOMIC_BEGIN:
    state.source.pc++;
    increment_active_atomic_number();
    break;
  case ATOMIC_END:
    decrement_active_atomic_number();
    state.source.pc++;

    // Don't context switch if the guard is false. This instruction hasn't
    // actually been executed, so context switching achieves nothing. (We
    // don't do this for the active_atomic_number though, because it's cheap,
    // and should be balanced under all circumstances anyway).
    if(!state.guard.is_false())
      force_cswitch();

    break;
  case RETURN:
    if(!state.guard.is_false())
    {
      expr2tc thecode = instruction.code, assign;
      if(make_return_assignment(assign, thecode))
        goto_symext::symex_assign(assign, true);
      symex_return(thecode);
      analyze_assign(assign);
    }
    state.source.pc++;
    break;
  default:
    goto_symext::symex_step(art);
  }
}

void execution_statet::symex_assign(
  const expr2tc &code,
  const bool hidden,
  const guardt &guard)
{
  pre_goto_guard = guardt();

  goto_symext::symex_assign(code, hidden, guard);

  if(threads_state.size() >= thread_cswitch_threshold)
    analyze_assign(code);
}

void execution_statet::claim(const expr2tc &expr, const std::string &msg)
{
  pre_goto_guard = guardt();

  goto_symext::claim(expr, msg);

  if(threads_state.size() >= thread_cswitch_threshold)
    analyze_read(expr);
}

void execution_statet::symex_goto(const expr2tc &old_guard)
{
  pre_goto_guard = threads_state[active_thread].guard;

  goto_symext::symex_goto(old_guard);

  if(!pre_goto_guard.is_false() && !is_nil_expr(old_guard))
  {
    if(threads_state.size() >= thread_cswitch_threshold)
    {
      analyze_read(old_guard);
    }
  }
}

void execution_statet::assume(const expr2tc &assumption)
{
  pre_goto_guard = guardt();

  goto_symext::assume(assumption);

  if(threads_state.size() >= thread_cswitch_threshold)
    analyze_read(assumption);
}

unsigned int &execution_statet::get_dynamic_counter()
{
  return dynamic_counter;
}

unsigned int &execution_statet::get_nondet_counter()
{
  return nondet_count;
}

goto_symex_statet &execution_statet::get_active_state()
{
  return threads_state.at(active_thread);
}

const goto_symex_statet &execution_statet::get_active_state() const
{
  return threads_state.at(active_thread);
}

unsigned int execution_statet::get_active_atomic_number()
{
  return atomic_numbers.at(active_thread);
}

void execution_statet::increment_active_atomic_number()
{
  atomic_numbers.at(active_thread)++;
}

void execution_statet::decrement_active_atomic_number()
{
  atomic_numbers.at(active_thread)--;
}

expr2tc execution_statet::get_guard_identifier()
{
  return symbol2tc(
    get_bool_type(),
    guard_execution,
    symbol2t::level1,
    CS_number,
    0,
    node_id,
    0);
}

void execution_statet::switch_to_thread(unsigned int i)
{
  last_active_thread = active_thread;
  active_thread = i;
  cur_state = &threads_state[active_thread];
}

bool execution_statet::dfs_explore_thread(unsigned int tid)
{
  if(DFS_traversed.at(tid))
    return false;

  if(threads_state.at(tid).call_stack.empty())
    return false;

  if(threads_state.at(tid).thread_ended)
    return false;

  DFS_traversed.at(tid) = true;
  return true;
}

bool execution_statet::check_if_ileaves_blocked()
{
  if(owning_rt->get_CS_bound() != -1 && CS_number >= owning_rt->get_CS_bound())
    return true;

  if(get_active_atomic_number() > 0)
    return true;

  if(owning_rt->directed_interleavings)
    // Don't generate interleavings automatically - instead, the user will
    // inserts intrinsics identifying where they want interleavings to occur,
    // and to what thread.
    return true;

  if(threads_state.size() < 2)
    return true;

  return false;
}

void execution_statet::end_thread()
{
  get_active_state().thread_ended = true;
  // If ending in an atomic block, the switcher fails to switch to another
  // live thread (because it's trying to be atomic). So, disable atomic blocks
  // when the thread ends.
  atomic_numbers[active_thread] = 0;
}

void execution_statet::update_after_switch_point()
{
  execute_guard();
  resetDFS_traversed();

  // MPOR records the variables accessed in last transition taken; we're
  // starting a new transition, so for the current thread, clear records.
  thread_last_reads[active_thread].clear();
  thread_last_writes[active_thread].clear();

  cswitch_forced = false;

  // If we've context switched, then wipe out all symbolic paths in the source
  // thread that didn't context switch, otherwise they'll observe other thread
  // PCs advancing with no change in state. However if we've hit a context
  // switch point and _not_ switched, don't wipe those symbolic paths, they
  // need to be preserved in at least one interleaving.
  if(last_active_thread != active_thread)
  {
    preserve_last_paths();
    cull_all_paths();
    restore_last_paths();
  }
}

void execution_statet::preserve_last_paths()
{
  // If the thread terminated, there are no paths to preserve: this is the final
  // switching away.
  if(threads_state[last_active_thread].thread_ended)
    return;

  // Examine the current execution state and the last insn, deciding which paths
  // are going to be preserved after this context switch. The current
  // instruction and guard are guaranteed (unless the guard is false), but if
  // we switched on a GOTO instruction, we may have forked. In that case we
  // need to find the branch that was generated there.

  auto &pp = preserved_paths[last_active_thread];
  auto &ls = threads_state[last_active_thread];
  assert(pp.size() == 0 && "Unmerged preserved paths in ex_state");
  assert(last_insn != nullptr && "Last insn unset in preserve_last_paths");

  // Add the current path to the set of paths to be preserved. Don't do this
  // if the current guard is false, though.
  if(!ls.guard.is_false())
    pp.push_back(std::make_pair(ls.source.pc, goto_statet(ls)));

  // Now then -- was it a goto? And did we actually branch to it? Detect this
  // by examining how the guard has changed: if there's no change, then the
  // GOTO condition must have evaluated to false.
  bool no_branch = (pre_goto_guard == ls.guard);
  if(last_insn->type == GOTO && !no_branch)
  {
    // We know where it branched to: fetch a reference to the list of all states
    // to be merged in there
    assert(last_insn->targets.size() == 1);
    auto target_insn_it = *last_insn->targets.begin();
    auto it = ls.top().goto_state_map.find(target_insn_it);
    assert(
      it != ls.top().goto_state_map.end() &&
      "Nonexistant preserved-path target?");
    auto &statelist = it->second;

    // There may be multiple paths in the map to be merged at that location,
    // for example if it's the loop end. Detect two circumstances: first where
    // the guard of the to-be-merged state is identical to the pre-goto guard,
    // meaning that the GOTO we executed had an unconditionally-true guard.
    // Second where the current-path guard plus the to-be-merged guard is equal
    // to the pre-goto guard: in that case, these can only be the two descendent
    // paths from the pre-goto state.
    const goto_statet *tomerge = nullptr;
    for(const goto_statet &gs : statelist)
    {
      bool merge = false;

      if(gs.guard == pre_goto_guard)
      {
        merge = true;
      }
      else
      {
        guardt tmp(ls.guard);
        tmp |= gs.guard;

        expr2tc foo = tmp.as_expr();
        expr2tc bar = pre_goto_guard.as_expr();
        do_simplify(foo);
        do_simplify(bar);

        if(foo == bar)
          merge = true;
      }

      // Select merging this goto_statet with a sanity check
      if(merge)
      {
        assert(tomerge == nullptr && "Multiple branching to-preserve paths?");
        tomerge = &gs;
      }
    }

    // We _must_ have found a path to merge, or the current-state guard would
    // have matched pre_goto_guard earlier
    assert(tomerge != nullptr);

    // Alas, copies.
    pp.emplace_back(std::make_pair(target_insn_it, goto_statet(*tomerge)));
  }

  // We must have picked up at least one path to merge
  if(pp.size() == 0)
  {
    // Even better: if the guard is now false, and we're context switching away,
    // then something like assume(0) occurred: no paths can continue in this
    // thread from this point. And anything we context switch to will get a
    // false guard too and thus expire.
    // Ideally at this point we would bail and return our formula to the RT
    // class, but that code is way too fragile. Instead, continue with an ended
    // thread that infects all other threads with it's false guard until we
    // complete.
    // It's unclear how to distinguish this case from an error in this code
    // here.
    // XXX methodise this
    threads_state[last_active_thread].thread_ended = true;
    atomic_numbers[last_active_thread] = 0;
  }
}

void execution_statet::cull_all_paths()
{
  // Walk through _all_ symbolic paths in the program and wipe them out.
  // Current path is easy: set the guard to false. phi_function will overwrite
  // any different valuation left in the l2 map.
  cur_state->guard.make_false();

  // This completely removes all symbolic paths that were going to be merged
  // back in at some point in the future.
  for(auto &frame : cur_state->call_stack)
  {
    frame.goto_state_map.clear();
  }
}

void execution_statet::restore_last_paths()
{
  // For each preserved path: create a fresh new goto_statet with data values
  // created from the present values of l2-renaming and value set, as we
  // (presumably) switch back in from a different thread. Then schedule the
  // states to be merged in at their original locations.
  // Given that we're discarding a lot of data here this could all be more
  // efficient, but it's what we've got.

  auto &list = preserved_paths[active_thread];
  for(auto const &p : list)
  {
    const auto &loc = p.first;
    const auto &gs = p.second;

    // Create a fresh new goto_statet to be merged in at the target insn
    assert(cur_state->top().goto_state_map[loc].size() == 0);
    cur_state->top().goto_state_map[loc].emplace_back(*cur_state);
    // Get ref to it
    auto &new_gs = *cur_state->top().goto_state_map[loc].begin();

    // Proceed to fill new_gs with old data. Ideally this would be a method...
    new_gs.depth = gs.depth;
    new_gs.guard = gs.guard;
    assert(new_gs.thread_id == gs.thread_id);

    // And that is it!
  }

  list.clear();
}

bool execution_statet::is_cur_state_guard_false()
{
  // So, can the assumption actually be true? If enabled, ask the solver.
  if(smt_thread_guard)
  {
    expr2tc parent_guard = threads_state[active_thread].guard.as_expr();

    runtime_encoded_equationt *rte =
      dynamic_cast<runtime_encoded_equationt *>(target.get());

    equality2tc the_question(gen_true_expr(), parent_guard);

    try
    {
      tvt res = rte->ask_solver_question(the_question);
      if(res.is_false())
        return true;
    }
    catch(runtime_encoded_equationt::dual_unsat_exception &e)
    {
      // Basically, this means our _assumptions_ here are false as well, so
      // neither true or false guards are possible. Consider this as meaning
      // that the guard is false.
      return true;
    }
  }

  return false;
}

void execution_statet::execute_guard()
{
  node_id = node_count++;
  expr2tc guard_expr = get_guard_identifier();
  expr2tc parent_guard;

  // Parent guard of this context switch - if a assign/claim/assume, just use
  // the current thread guard. However if we just executed a goto, use the
  // pre-goto thread guard that we stored at that time. This is so that the
  // thread we context switch to gets the guard of that context switch happening
  // rather than the guard of either branch of the GOTO.
  if(!pre_goto_guard.is_true())
    parent_guard = pre_goto_guard.as_expr();
  else
    parent_guard = threads_state[last_active_thread].guard.as_expr();

  // Rename value, allows its use in other renamed exprs
  state_level2->make_assignment(guard_expr, expr2tc(), expr2tc());

  // Truth of this guard implies the parent is true.
  state_level2->rename(parent_guard);
  do_simplify(parent_guard);
  implies2tc assumpt(guard_expr, parent_guard);

  target->assumption(
    guardt().as_expr(), assumpt, get_active_state().source, first_loop);

  guardt old_guard;
  old_guard.add(threads_state[last_active_thread].guard.as_expr());

  // If we simplified the global guard expr to false, write that to thread
  // guards, not the symbolic guard name. This is the only way to bail out of
  // evaulating a particular interleaving early right now.
  if(is_false(parent_guard))
    guard_expr = parent_guard;

  for(auto &i : threads_state)
  {
    i.global_guard.make_true();
    i.global_guard.add(get_guard_identifier());
  }

  // Check to see whether or not the state guard is false, indicating we've
  // found an unviable interleaving. However don't do this if we didn't
  // /actually/ switch between threads, because it's acceptable to have a
  // context switch point in a branch where the guard is false (it just isn't
  // acceptable to permit switching).
  if(last_active_thread != active_thread && is_cur_state_guard_false())
    interleaving_unviable = true;
}

unsigned int execution_statet::add_thread(const goto_programt *prog)
{
  goto_symex_statet new_state(*state_level2, global_value_set, ns);
  new_state.initialize(
    prog->instructions.begin(),
    prog->instructions.end(),
    prog,
    threads_state.size());

  unsigned int thread_nr = threads_state.size();
  new_state.source.thread_nr = thread_nr;
  new_state.global_guard.make_true();
  new_state.global_guard.add(get_guard_identifier());
  threads_state.push_back(new_state);
  preserved_paths.emplace_back();
  atomic_numbers.push_back(0);

  if(DFS_traversed.size() <= new_state.source.thread_nr)
  {
    DFS_traversed.push_back(false);
  }
  else
  {
    DFS_traversed[new_state.source.thread_nr] = false;
  }

  thread_start_data.emplace_back();

  // We invalidated all threads_state refs, so reset cur_state ptr.
  cur_state = &threads_state[active_thread];

  // Update MPOR tracking data with newly initialized thread
  thread_last_reads.emplace_back();
  thread_last_writes.emplace_back();
  // Unfortunately as each thread has a depenancy relation with every other
  // thread we have to do a lot of work to initialize a new one. And initially
  // all relations are '0', no transitions yet.
  for(auto &it : dependancy_chain)
  {
    it.push_back(0);
  }
  // And the new threads dependancies,
  dependancy_chain.emplace_back();
  for(unsigned int i = 0; i < dependancy_chain.size(); i++)
    dependancy_chain.back().push_back(0);

  // While we've recorded the new thread as starting in the designated program,
  // it might not run immediately, thus must have it's path preserved:
  preserved_paths[thread_nr].push_back(std::make_pair(
    prog->instructions.begin(), goto_statet(threads_state[thread_nr])));

  return threads_state.size() - 1; // thread ID, zero based
}

void execution_statet::analyze_assign(const expr2tc &code)
{
  if(is_nil_expr(code))
    return;

  std::set<expr2tc> global_reads, global_writes;
  const code_assign2t &assign = to_code_assign2t(code);
  get_expr_globals(ns, assign.target, global_writes);
  get_expr_globals(ns, assign.source, global_reads);

  if(global_reads.size() > 0 || global_writes.size() > 0)
  {
    // Record read/written data
    thread_last_reads[active_thread].insert(
      global_reads.begin(), global_reads.end());
    thread_last_writes[active_thread].insert(
      global_writes.begin(), global_writes.end());
  }
}

void execution_statet::analyze_read(const expr2tc &code)
{
  std::set<expr2tc> global_reads, global_writes;
  get_expr_globals(ns, code, global_reads);

  if(global_reads.size() > 0)
  {
    // Record read/written data
    thread_last_reads[active_thread].insert(
      global_reads.begin(), global_reads.end());
  }
}

void execution_statet::get_expr_globals(
  const namespacet &ns,
  const expr2tc &expr,
  std::set<expr2tc> &globals_list)
{
  if(is_nil_expr(expr))
    return;

  if(
    is_address_of2t(expr) || is_pointer_type(expr) || is_valid_object2t(expr) ||
    is_dynamic_size2t(expr))
  {
    return;
  }
  if(is_symbol2t(expr))
  {
    expr2tc newexpr = expr;
    get_active_state().get_original_name(newexpr);
    const std::string &name = to_symbol2t(newexpr).thename.as_string();

    if(
      name == "goto_symex::guard!" +
                i2string(get_active_state().top().level1.thread_id))
      return;

    const symbolt *symbol;
    if(ns.lookup(name, symbol))
      return;

    if(
      name == "c:@__ESBMC_alloc" || name == "c:@__ESBMC_alloc_size" ||
      name == "c:@__ESBMC_is_dynamic")
    {
      return;
    }
    if((symbol->static_lifetime || symbol->type.is_dynamic_set()))
    {
      std::list<unsigned int> threadId_list;
      std::map<expr2tc, std::list<unsigned int>>::iterator it_find;
      it_find = vars_map.find(expr);

      //the expression was accessed in another interleaving
      if(it_find != vars_map.end())
      {
        threadId_list = it_find->second;
        threadId_list.push_back(get_active_state().top().level1.thread_id);

        vars_map.insert(
          std::pair<expr2tc, std::list<unsigned int>>(expr, threadId_list));

        std::list<unsigned int>::iterator it_list;
        for(it_list = threadId_list.begin(); it_list != threadId_list.end();
            ++it_list)
        {
          //find if some thread access the same expression
          if(*it_list != get_active_state().top().level1.thread_id)
          {
            globals_list.insert(expr);
            is_global.insert(std::pair<expr2tc, bool>(expr, true));
          }
          //expression was not accessed by other thread
          else
          {
            std::map<expr2tc, bool>::iterator its_global;
            its_global = is_global.find(expr);
            //expression was defined as global in another interleaving
            if(its_global != is_global.end())
            {
              globals_list.insert(expr);
            }
          }
        }
        //first access of expression
      }
      else
      {
        std::map<expr2tc, bool>::iterator its_global;
        its_global = is_global.find(expr);
        if(its_global != is_global.end())
        {
          globals_list.insert(expr);
        }
        else
        {
          threadId_list.push_back(get_active_state().top().level1.thread_id);
          vars_map.insert(
            std::pair<expr2tc, std::list<unsigned int>>(expr, threadId_list));
          globals_list.insert(expr);
        }
      }
    }
    else
    {
      return;
    }
  }

  expr->foreach_operand([this, &globals_list, &ns](const expr2tc &e) {
    get_expr_globals(ns, e, globals_list);
  });
}

bool execution_statet::check_mpor_dependancy(unsigned int j, unsigned int l)
  const
{
  assert(j < threads_state.size());
  assert(l < threads_state.size());

  // Rules given on page 13 of MPOR paper, although they don't appear to
  // distinguish which thread is which correctly. Essentially, check that
  // the write(s) of the previous transition (l) don't intersect with this
  // transitions (j) reads or writes; and that the previous transitions reads
  // don't intersect with this transitions write(s).

  // Double write intersection
  for(std::set<expr2tc>::const_iterator it = thread_last_writes[j].begin();
      it != thread_last_writes[j].end();
      it++)
    if(thread_last_writes[l].find(*it) != thread_last_writes[l].end())
      return true;

  // This read what that wrote intersection
  for(std::set<expr2tc>::const_iterator it = thread_last_reads[j].begin();
      it != thread_last_reads[j].end();
      it++)
    if(thread_last_writes[l].find(*it) != thread_last_writes[l].end())
      return true;

  // We wrote what that reads intersection
  for(std::set<expr2tc>::const_iterator it = thread_last_writes[j].begin();
      it != thread_last_writes[j].end();
      it++)
    if(thread_last_reads[l].find(*it) != thread_last_reads[l].end())
      return true;

  // No check for read-read intersection, it doesn't affect anything
  return false;
}

void execution_statet::calculate_mpor_constraints()
{
  // Primary bit of MPOR logic - to be executed at the end of a transition to
  // update dependancy tracking and suchlike.

  // MPOR paper, page 12, create new dependancy chain record for this time step.

  // 2D Vector of relations, T x T, i.e. threads on each axis:
  //    -1 signifies that no relation exists.
  //    0 that the thread hasn't run yet.
  //    1 that there is a dependency between these threads.
  //
  //  dependancy_chain contains the state from the previous transition taken;
  //  here we update it to reflect the latest transition, and make a decision
  //  about progress later.
  std::vector<std::vector<int>> new_dep_chain = dependancy_chain;

  // Start new dependancy chain for this thread. Default to there being no
  // relation.
  for(unsigned int i = 0; i < new_dep_chain.size(); i++)
    new_dep_chain[active_thread][i] = -1;

  // This thread depends on this thread.
  new_dep_chain[active_thread][active_thread] = 1;

  // Mark un-run threads as continuing to be un-run. Otherwise, look for a
  // dependancy chain from each thread to the run thread.
  for(unsigned int j = 0; j < new_dep_chain.size(); j++)
  {
    if(j == active_thread)
      continue;

    if(dependancy_chain[j][active_thread] == 0)
    {
      // This thread hasn't been run; continue not having been run.
      new_dep_chain[j][active_thread] = 0;
    }
    else
    {
      // This is where the beef is. If there is any other thread (including
      // the active thread) that we depend on, that depends on the active
      // thread, then record a dependancy.
      // A direct dependancy occurs when l = j, as DCjj always = 1, and DEPji
      // is true.
      int res = 0;

      for(unsigned int l = 0; l < new_dep_chain.size(); l++)
      {
        if(dependancy_chain[j][l] != 1)
          continue; // No dependancy relation here

        // Now check for variable dependancy.
        if(!check_mpor_dependancy(active_thread, l))
          continue;

        res = 1;
        break;
      }

      // Don't overwrite if no match
      if(res != 0)
        new_dep_chain[j][active_thread] = res;
    }
  }

  // For /all other relations/, just propagate the dependancy it already has.
  // Achieved by initial duplication of dependancy_chain.

  // Voila, new dependancy chain.

  // Calculate whether or not the transition we just took, in active_thread,
  // was in fact schedulable. We can't tell whether or not a transition is
  // allowed in advance because we don't know what it is. So instead, check
  // whether or not a transition /would/ have been allowed, once we've taken
  // it.
  bool can_run = true;
  for(unsigned int j = active_thread + 1; j < threads_state.size(); j++)
  {
    if(new_dep_chain[j][active_thread] != -1)
      // Either no higher threads have been run, or a dependancy relation in
      // a higher thread justifies our out-of-order execution.
      continue;

    // Search for a dependancy chain in a lower thread that links us back to
    // a higher thread, justifying this order.
    bool dep_exists = false;
    for(unsigned int l = 0; l < active_thread; l++)
    {
      if(dependancy_chain[j][l] == 1)
        dep_exists = true;
    }

    if(!dep_exists)
    {
      can_run = false;
      break;
    }
  }

  mpor_says_no = !can_run;

  dependancy_chain = new_dep_chain;
}

bool execution_statet::has_cswitch_point_occured() const
{
  // Context switches can occur due to being forced, or by global state access

  if(cswitch_forced)
    return true;

  if(
    thread_last_reads[active_thread].size() != 0 ||
    thread_last_writes[active_thread].size() != 0)
    return true;

  return false;
}

bool execution_statet::can_execution_continue() const
{
  if(threads_state[active_thread].thread_ended)
    return false;

  if(threads_state[active_thread].call_stack.empty())
    return false;

  return true;
}

crypto_hash execution_statet::generate_hash() const
{
  auto l2 = boost::dynamic_pointer_cast<state_hashing_level2t>(state_level2);
  assert(l2 != nullptr);

  crypto_hash state = l2->generate_l2_state_hash();
  std::string str = state.to_string();

  for(const auto &it : threads_state)
  {
    goto_programt::const_targett pc = it.source.pc;
    int id = pc->location_number;
    std::stringstream s;
    s << id;
    str += "!" + s.str();
  }

  crypto_hash h;
  h.ingest(str.c_str(), str.size());
  h.fin();

  return h;
}

crypto_hash execution_statet::update_hash_for_assignment(const expr2tc &rhs)
{
  crypto_hash h;
  rhs->hash(h);
  h.fin();
  return h;
}

void execution_statet::print_stack_traces(unsigned int indent) const
{
  std::vector<goto_symex_statet>::const_iterator it;
  std::string spaces = std::string("");
  unsigned int i;

  for(i = 0; i < indent; i++)
    spaces += " ";

  i = 0;
  for(it = threads_state.begin(); it != threads_state.end(); it++)
  {
    std::cout << spaces << "Thread " << i++ << ":" << std::endl;
    it->print_stack_trace(indent + 2);
    std::cout << std::endl;
  }
}

void execution_statet::switch_to_monitor()
{
  if(threads_state[monitor_tid].thread_ended)
  {
    if(!mon_thread_warning)
    {
      std::cerr << "Switching to ended monitor; you need to increase its "
                   "context or prefix bound"
                << std::endl;
      mon_thread_warning = true;
    }

    return;
  }

  assert(tid_is_set && "Must set monitor thread before switching to monitor\n");
  assert(
    !mon_from_tid && "Switching to monitor without having switched away\n");

  monitor_from_tid = active_thread;
  mon_from_tid = true;

  if(monitor_tid != get_active_state_number())
  {
    // Don't call switch_to_thread -- it'll execute the thread guard, which is
    // an extremely bad plan.
    last_active_thread = active_thread;
    active_thread = monitor_tid;
    cur_state = &threads_state[active_thread];
    cur_state->guard = threads_state[last_active_thread].guard;
  }
  else
  {
    assert(0 && "Switching to monitor thread from self\n");
  }
}

void execution_statet::switch_away_from_monitor()
{
  // Occurs when we rerun the automata to discover whether or not the property
  // has been violated or not.
  if(threads_state[monitor_tid].thread_ended)
    return;

  assert(tid_is_set && "Must set monitor thread before switching from mon\n");
  assert(mon_from_tid && "Switching from monitor without switching to\n");

  assert(
    monitor_tid == active_thread &&
    "Must call switch_from_monitor from monitor thread\n");

  // Don't call switch_to_thread -- it'll execute the thread guard, which is
  // an extremely bad plan.
  last_active_thread = active_thread;
  active_thread = monitor_from_tid;
  cur_state = &threads_state[active_thread];

  cur_state->guard = threads_state[monitor_tid].guard;

  mon_from_tid = false;
}

void execution_statet::kill_monitor_thread()
{
  assert(
    monitor_tid != active_thread &&
    "You cannot kill monitor thread _from_ the monitor thread\n");

  threads_state[monitor_tid].thread_ended = true;
}

execution_statet::ex_state_level2t::ex_state_level2t(execution_statet &ref)
  : owner(&ref)
{
}

boost::shared_ptr<renaming::level2t>
execution_statet::ex_state_level2t::clone() const
{
  return boost::shared_ptr<ex_state_level2t>(new ex_state_level2t(*this));
}

void execution_statet::ex_state_level2t::rename(
  expr2tc &lhs_sym,
  unsigned count)
{
  renaming::level2t::coveredinbees(lhs_sym, count, owner->node_id);
}

void execution_statet::ex_state_level2t::rename(expr2tc &identifier)
{
  renaming::level2t::rename(identifier);
}

dfs_execution_statet::~dfs_execution_statet()
{
  // Delete target; or if we're encoding at runtime, pop a context.
  if(smt_during_symex)
    target->pop_ctx();
}

boost::shared_ptr<execution_statet> dfs_execution_statet::clone() const
{
  boost::shared_ptr<dfs_execution_statet> d =
    boost::shared_ptr<dfs_execution_statet>(new dfs_execution_statet(*this));

  // Duplicate target equation; or if we're encoding at runtime, push a context.
  if(smt_during_symex)
  {
    d->target = target;
    d->target->push_ctx();
  }
  else
  {
    d->target = target->clone();
  }

  return d;
}

schedule_execution_statet::~schedule_execution_statet()
{
  // Don't delete equation. Schedule requires all this data.
}

boost::shared_ptr<execution_statet> schedule_execution_statet::clone() const
{
  boost::shared_ptr<schedule_execution_statet> s =
    boost::shared_ptr<schedule_execution_statet>(
      new schedule_execution_statet(*this));

  // Don't duplicate target equation.
  s->target = target;
  return s;
}

void schedule_execution_statet::claim(
  const expr2tc &expr,
  const std::string &msg)
{
  unsigned int tmp_total, tmp_remaining;

  tmp_total = total_claims;
  tmp_remaining = remaining_claims;

  execution_statet::claim(expr, msg);

  tmp_total = total_claims - tmp_total;
  tmp_remaining = remaining_claims - tmp_remaining;

  *ptotal_claims += tmp_total;
  *premaining_claims += tmp_remaining;
}

execution_statet::state_hashing_level2t::state_hashing_level2t(
  execution_statet &ref)
  : ex_state_level2t(ref)
{
}

boost::shared_ptr<renaming::level2t>
execution_statet::state_hashing_level2t::clone() const
{
  return boost::shared_ptr<state_hashing_level2t>(
    new state_hashing_level2t(*this));
}

void execution_statet::state_hashing_level2t::make_assignment(
  expr2tc &lhs_sym,
  const expr2tc &const_value,
  const expr2tc &assigned_value)
{
  //  crypto_hash hash;

  renaming::level2t::make_assignment(lhs_sym, const_value, assigned_value);

  // If there's no body to the assignment, don't hash.
  if(!is_nil_expr(assigned_value))
  {
    // XXX - consider whether to use l1 names instead. Recursion, reentrancy.
    crypto_hash hash = owner->update_hash_for_assignment(assigned_value);
    std::string orig_name = to_symbol2t(lhs_sym).thename.as_string();
    current_hashes[orig_name] = hash;
  }
}

crypto_hash
execution_statet::state_hashing_level2t::generate_l2_state_hash() const
{
  unsigned int total;

  uint8_t *data = (uint8_t *)alloca(
    current_hashes.size() * CRYPTO_HASH_SIZE * sizeof(uint8_t));

  total = 0;
  for(const auto &current_hashe : current_hashes)
  {
    memcpy(
      &data[total * CRYPTO_HASH_SIZE],
      current_hashe.second.hash,
      CRYPTO_HASH_SIZE);
    total++;
  }

  crypto_hash c;
  c.ingest(data, total * CRYPTO_HASH_SIZE);
  c.fin();
  return c;
}
