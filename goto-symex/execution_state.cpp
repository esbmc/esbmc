/*******************************************************************\

   Module:

   Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>
#include <langapi/mode.h>
#include <langapi/languages.h>
#include <langapi/language_ui.h>

#include "execution_state.h"
#include "reachability_tree.h"
#include <string>
#include <sstream>
#include <vector>
#include <i2string.h>
#include <string2array.h>
#include <std_expr.h>
#include <expr_util.h>
#include <c_types.h>
#include <simplify_expr.h>
#include "config.h"

unsigned int execution_statet::node_count = 0;
std::map<expr2tc, std::list<unsigned int>> vars_map;
std::map<expr2tc, bool> is_global;

execution_statet::execution_statet(const goto_functionst &goto_functions,
                                   const namespacet &ns,
                                   reachability_treet *art,
                                   std::shared_ptr<symex_targett> _target,
                                   contextt &context,
                                   std::shared_ptr<ex_state_level2t> l2init,
                                   optionst &options,
                                   message_handlert &_message_handler) :
  goto_symext(ns, context, goto_functions, _target, options),
  owning_rt(art),
  state_level2(l2init),
  global_value_set(ns),
  message_handler(_message_handler)
{

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
    goto_functions.function_map.find("main");
  if (it == goto_functions.function_map.end()) {
    std::cerr << "main symbol not found; please set an entry point" <<std::endl;
    abort();
  }

  const goto_programt *goto_program = &(it->second.body);

  // Initialize initial thread state
  goto_symex_statet state(*state_level2, global_value_set, ns);
  state.initialize((*goto_program).instructions.begin(),
             (*goto_program).instructions.end(),
             goto_program, 0);

  threads_state.push_back(state);
  cur_state = &threads_state.front();
  cur_state->global_guard.make_true();
  cur_state->global_guard.add(get_guard_identifier());

  atomic_numbers.push_back(0);

  if (DFS_traversed.size() <= state.source.thread_nr) {
    DFS_traversed.push_back(false);
  } else {
    DFS_traversed[state.source.thread_nr] = false;
  }

  thread_start_data.push_back(expr2tc());

  // Initial mpor tracking.
  thread_last_reads.push_back(std::set<expr2tc>());
  thread_last_writes.push_back(std::set<expr2tc>());
  // One thread with one dependency relation.
  dependancy_chain.push_back(std::vector<int>());
  dependancy_chain.back().push_back(0);
  mpor_says_no = false;

  cswitch_forced = false;
  active_thread = 0;
  last_active_thread = 0;
  node_count = 0;
  nondet_count = 0;
  dynamic_counter = 0;
  DFS_traversed.reserve(1);
  DFS_traversed[0] = false;
  check_ltl = false;
  mon_thread_warning = false;

  bool ltl_mode =
    (options.get_bool_option("ltl") || options.get_bool_option("ltl-dummy"));
  thread_cswitch_threshold = (ltl_mode) ? 3 : 2;
}

execution_statet::execution_statet(const execution_statet &ex) :
  goto_symext(ex),
  owning_rt(ex.owning_rt),
  state_level2(std::dynamic_pointer_cast<ex_state_level2t>(ex.state_level2->clone())),
  global_value_set(ex.global_value_set),
  message_handler(ex.message_handler)
{

  *this = ex;

  // Regenerate threads state using new objects state_level2 ref
  threads_state.clear();
  std::vector<goto_symex_statet>::const_iterator it;
  for (it = ex.threads_state.begin(); it != ex.threads_state.end(); it++) {
    goto_symex_statet state(*it, *state_level2, global_value_set);
    threads_state.push_back(state);
  }

  // Reassign which state is currently being worked on.
  cur_state = &threads_state[active_thread];
}

execution_statet&
execution_statet::operator=(const execution_statet &ex)
{
  // Don't copy level2, copy cons it in execution_statet(ref)
  //state_level2 = ex.state_level2;

  threads_state = ex.threads_state;
  atomic_numbers = ex.atomic_numbers;
  DFS_traversed = ex.DFS_traversed;
  thread_start_data = ex.thread_start_data;
  last_active_thread = ex.last_active_thread;
  active_thread = ex.active_thread;
  guard_execution = ex.guard_execution;
  nondet_count = ex.nondet_count;
  dynamic_counter = ex.dynamic_counter;
  node_id = ex.node_id;
  global_value_set = ex.global_value_set;
  interleaving_unviable = ex.interleaving_unviable;
  pre_goto_guard = ex.pre_goto_guard;
  mon_thread_warning = ex.mon_thread_warning;
  check_ltl = ex.check_ltl;
  property_monitor_strings = ex.property_monitor_strings;

  monitor_tid = ex.monitor_tid;
  tid_is_set = ex.tid_is_set;
  monitor_from_tid = ex.monitor_from_tid;
  mon_from_tid = ex.mon_from_tid;
  thread_cswitch_threshold = ex.thread_cswitch_threshold;
  symex_trace = ex.symex_trace;
  smt_during_symex = ex.smt_during_symex;
  smt_thread_guard = ex.smt_thread_guard;

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
  for (std::vector<goto_symex_statet>::iterator it = threads_state.begin();
       it != threads_state.end(); it++) {
    for (goto_symex_statet::call_stackt::iterator it2 = it->call_stack.begin();
         it2 != it->call_stack.end(); it2++) {
      for (goto_symex_statet::goto_state_mapt::iterator it3 = it2->goto_state_map.begin();
           it3 != it2->goto_state_map.end(); it3++) {
        for (goto_symex_statet::goto_state_listt::iterator it4 = it3->second.begin();
             it4 != it3->second.begin(); it4++) {
          ex_state_level2t &l2 = dynamic_cast<ex_state_level2t&>(it4->level2);
          l2.owner = this;
        }
      }
    }
  }

  state_level2->owner = this;

  return *this;
}

execution_statet::~execution_statet()
{
};

void
execution_statet::symex_step(reachability_treet &art)
{

  statet &state = get_active_state();
  const goto_programt::instructiont &instruction = *state.source.pc;

  merge_gotos();

  if (break_insn != 0 && break_insn == instruction.location_number) {
#ifndef _WIN32
    __asm__("int $3");
#else
    std::cerr << "Can't trap on windows, sorry" << std::endl;
    abort();
#endif
  }

  if (symex_trace) {
    const goto_programt p_dummy;
    goto_functionst::function_mapt::const_iterator it =
      goto_functions.function_map.find(instruction.function);

    const goto_programt &p_real = it->second.body;
    const goto_programt &p = (it == goto_functions.function_map.end()) ? p_dummy : p_real;
    p.output_instruction(ns, "", std::cout, state.source.pc, false, false);
  }

  switch (instruction.type) {
    case END_FUNCTION:
      if (instruction.function == "main") {
        end_thread();
        force_cswitch();
      } else {
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
      if (!state.guard.is_false())
        force_cswitch();

      break;
    case RETURN:
      if(!state.guard.is_false()) {
        expr2tc thecode = instruction.code, assign;
        if (make_return_assignment(assign, thecode)) {
          goto_symext::symex_assign(assign);
        }

        symex_return();

        if (!is_nil_expr(assign))
          analyze_assign(assign);
      }
      state.source.pc++;
      break;
    default:
      goto_symext::symex_step(art);
  }

  return;
}

void
execution_statet::symex_assign(const expr2tc &code)
{
  pre_goto_guard = expr2tc();

  goto_symext::symex_assign(code);

  if (threads_state.size() >= thread_cswitch_threshold)
    analyze_assign(code);

  return;
}

void
execution_statet::claim(const expr2tc &expr, const std::string &msg)
{
  pre_goto_guard = expr2tc();

  goto_symext::claim(expr, msg);

  if (threads_state.size() >= thread_cswitch_threshold)
    analyze_read(expr);

  return;
}

void
execution_statet::symex_goto(const expr2tc &old_guard)
{
  pre_goto_guard = threads_state[active_thread].guard.as_expr();

  goto_symext::symex_goto(old_guard);

  if (!is_nil_expr(old_guard)) {
    if (threads_state.size() >= thread_cswitch_threshold) {
      analyze_read(old_guard);
    }
  }

  return;
}

void
execution_statet::assume(const expr2tc &assumption)
{
  pre_goto_guard = expr2tc();

  goto_symext::assume(assumption);

  if (threads_state.size() >= thread_cswitch_threshold)
    analyze_read(assumption);

  return;
}

unsigned int &
execution_statet::get_dynamic_counter(void)
{

  return dynamic_counter;
}

unsigned int &
execution_statet::get_nondet_counter(void)
{

  return nondet_count;
}

goto_symex_statet &
execution_statet::get_active_state() {

  return threads_state.at(active_thread);
}

const goto_symex_statet &
execution_statet::get_active_state() const
{
  return threads_state.at(active_thread);
}

unsigned int
execution_statet::get_active_atomic_number()
{

  return atomic_numbers.at(active_thread);
}

void
execution_statet::increment_active_atomic_number()
{

  atomic_numbers.at(active_thread)++;
}

void
execution_statet::decrement_active_atomic_number()
{

  atomic_numbers.at(active_thread)--;
}

expr2tc
execution_statet::get_guard_identifier()
{

  return symbol2tc(get_bool_type(), guard_execution, symbol2t::level1,
                   CS_number, 0, node_id, 0);
}

void
execution_statet::switch_to_thread(unsigned int i)
{

  last_active_thread = active_thread;
  active_thread = i;
  cur_state = &threads_state[active_thread];
}

bool
execution_statet::dfs_explore_thread(unsigned int tid)
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

bool
execution_statet::check_if_ileaves_blocked(void)
{

  if(owning_rt->get_CS_bound() != -1 && CS_number >= owning_rt->get_CS_bound())
    return true;

  if (get_active_atomic_number() > 0)
    return true;

  if (owning_rt->directed_interleavings)
    // Don't generate interleavings automatically - instead, the user will
    // inserts intrinsics identifying where they want interleavings to occur,
    // and to what thread.
    return true;

  if(threads_state.size() < 2)
    return true;

  return false;
}

void
execution_statet::end_thread(void)
{

  get_active_state().thread_ended = true;
  // If ending in an atomic block, the switcher fails to switch to another
  // live thread (because it's trying to be atomic). So, disable atomic blocks
  // when the thread ends.
  atomic_numbers[active_thread] = 0;
}

void
execution_statet::update_after_switch_point(void)
{

  execute_guard();
  resetDFS_traversed();

  // MPOR records the variables accessed in last transition taken; we're
  // starting a new transition, so for the current thread, clear records.
  thread_last_reads[active_thread].clear();
  thread_last_writes[active_thread].clear();

  cswitch_forced = false;
}

bool
execution_statet::is_cur_state_guard_false(void)
{

  // So, can the assumption actually be true? If enabled, ask the solver.
  if (smt_thread_guard) {
    expr2tc parent_guard = threads_state[active_thread].guard.as_expr();

    runtime_encoded_equationt *rte = dynamic_cast<runtime_encoded_equationt*>
                                                 (target.get());

    equality2tc the_question(true_expr, parent_guard);

    try {
      tvt res = rte->ask_solver_question(the_question);
      if (res.is_false())
        return true;
    } catch (runtime_encoded_equationt::dual_unsat_exception &e) {
      // Basically, this means our _assumptions_ here are false as well, so
      // neither true or false guards are possible. Consider this as meaning
      // that the guard is false.
      return true;
    }
  }

  return false;
}

void
execution_statet::execute_guard(void)
{

  node_id = node_count++;
  expr2tc guard_expr = get_guard_identifier();
  exprt new_rhs, const_prop_val;
  expr2tc parent_guard;

  // Parent guard of this context switch - if a assign/claim/assume, just use
  // the current thread guard. However if we just executed a goto, use the
  // pre-goto thread guard that we stored at that time. This is so that the
  // thread we context switch to gets the guard of that context switch happening
  // rather than the guard of either branch of the GOTO.
  if (!is_nil_expr(pre_goto_guard))
    parent_guard = pre_goto_guard;
  else
    parent_guard = threads_state[last_active_thread].guard.as_expr();

  // Rename value, allows its use in other renamed exprs
  state_level2->make_assignment(guard_expr, expr2tc(), expr2tc());

  // Truth of this guard implies the parent is true.
  state_level2->rename(parent_guard);
  do_simplify(parent_guard);
  implies2tc assumpt(guard_expr, parent_guard);

  target->assumption(guardt().as_expr(), assumpt, get_active_state().source);

  guardt old_guard;
  old_guard.add(threads_state[last_active_thread].guard.as_expr());

  // If we simplified the global guard expr to false, write that to thread
  // guards, not the symbolic guard name. This is the only way to bail out of
  // evaulating a particular interleaving early right now.
  if (is_false(parent_guard))
    guard_expr = parent_guard;

  for (unsigned int i = 0; i < threads_state.size(); i++)
  {
    threads_state.at(i).global_guard.make_true();
    threads_state.at(i).global_guard.add(get_guard_identifier());
  }

  // Check to see whether or not the state guard is false, indicating we've
  // found an unviable interleaving. However don't do this if we didn't
  // /actually/ switch between threads, because it's acceptable to have a
  // context switch point in a branch where the guard is false (it just isn't
  // acceptable to permit switching).
  if (last_active_thread != active_thread && is_cur_state_guard_false())
    interleaving_unviable = true;
}

unsigned int
execution_statet::add_thread(const goto_programt *prog)
{

  goto_symex_statet new_state(*state_level2, global_value_set, ns);
  new_state.initialize(prog->instructions.begin(), prog->instructions.end(),
                      prog, threads_state.size());

  new_state.source.thread_nr = threads_state.size();
  new_state.global_guard.make_true();
  new_state.global_guard.add(get_guard_identifier());
  threads_state.push_back(new_state);
  atomic_numbers.push_back(0);

  if (DFS_traversed.size() <= new_state.source.thread_nr) {
    DFS_traversed.push_back(false);
  } else {
    DFS_traversed[new_state.source.thread_nr] = false;
  }

  thread_start_data.push_back(expr2tc());

  // We invalidated all threads_state refs, so reset cur_state ptr.
  cur_state = &threads_state[active_thread];

  // Update MPOR tracking data with newly initialized thread
  thread_last_reads.push_back(std::set<expr2tc>());
  thread_last_writes.push_back(std::set<expr2tc>());
  // Unfortunately as each thread has a depenancy relation with every other
  // thread we have to do a lot of work to initialize a new one. And initially
  // all relations are '0', no transitions yet.
  for (std::vector<std::vector<int> >::iterator it = dependancy_chain.begin();
       it != dependancy_chain.end(); it++) {
    it->push_back(0);
  }
  // And the new threads dependancies,
  dependancy_chain.push_back(std::vector<int>());
  for (unsigned int i = 0; i < dependancy_chain.size(); i++)
    dependancy_chain.back().push_back(0);

  return threads_state.size() - 1; // thread ID, zero based
}

void
execution_statet::analyze_assign(const expr2tc &code)
{

  std::set<expr2tc> global_reads, global_writes;
  const code_assign2t &assign = to_code_assign2t(code);
  get_expr_globals(ns, assign.target, global_writes);
  get_expr_globals(ns, assign.source, global_reads);

  if (global_reads.size() > 0 || global_writes.size() > 0) {
    // Record read/written data
    thread_last_reads[active_thread].insert(global_reads.begin(),
                                            global_reads.end());
    thread_last_writes[active_thread].insert(global_writes.begin(),
                                             global_writes.end());
  }

  return;
}

void
execution_statet::analyze_read(const expr2tc &code)
{

  std::set<expr2tc> global_reads, global_writes;
  get_expr_globals(ns, code, global_reads);

  if (global_reads.size() > 0) {
    // Record read/written data
    thread_last_reads[active_thread].insert(global_reads.begin(),
                                            global_reads.end());
  }

  return;
}

void
execution_statet::get_expr_globals(const namespacet &ns, const expr2tc &expr,
                                   std::set<expr2tc> &globals_list)
{

  if (is_nil_expr(expr))
    return;

  if (is_address_of2t(expr) || is_pointer_type(expr) ||
      is_valid_object2t(expr) || is_dynamic_size2t(expr) ||
      is_dynamic_size2t(expr)) {
    return;
  } else if (is_symbol2t(expr)) {
    expr2tc newexpr = expr;
    get_active_state().get_original_name(newexpr);
    const std::string &name = to_symbol2t(newexpr).thename.as_string();

    if (name == "goto_symex::guard!" +
        i2string(get_active_state().top().level1.thread_id))
      return;

    const symbolt *symbol;
    if (ns.lookup(name, symbol))
      return;

    if (name == "c::__ESBMC_alloc" || name == "c::__ESBMC_alloc_size" ||
        name == "c::__ESBMC_is_dynamic") {
      return;
    }
    else if ((symbol->static_lifetime || symbol->type.is_dynamic_set()))
    {
      std::list<unsigned int> threadId_list;
      std::map<expr2tc, std::list<unsigned int>>::iterator it_find;
      it_find = vars_map.find(expr);

      //the expression was accessed in another interleaving
      if (it_find != vars_map.end())
      {
        threadId_list = it_find->second;
        threadId_list.push_back(get_active_state().top().level1.thread_id);

        vars_map.insert(
          std::pair<expr2tc, std::list<unsigned int>>(expr, threadId_list));

        std::list<unsigned int>::iterator it_list;
        for (it_list = threadId_list.begin(); it_list != threadId_list.end();
            ++it_list)
        {

          //find if some thread access the same expression
          if (*it_list != get_active_state().top().level1.thread_id)
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
            if (its_global != is_global.end())
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
        if (its_global != is_global.end())
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
    } else {
      return;
    }
  }

  expr->foreach_operand([this, &globals_list, &ns] (const expr2tc &e) {
    get_expr_globals(ns, e, globals_list);
    }
  );
}

bool
execution_statet::check_mpor_dependancy(unsigned int j, unsigned int l) const
{

  assert(j < threads_state.size());
  assert(l < threads_state.size());

  // Rules given on page 13 of MPOR paper, although they don't appear to
  // distinguish which thread is which correctly. Essentially, check that
  // the write(s) of the previous transition (l) don't intersect with this
  // transitions (j) reads or writes; and that the previous transitions reads
  // don't intersect with this transitions write(s).

  // Double write intersection
  for (std::set<expr2tc>::const_iterator it = thread_last_writes[j].begin();
       it != thread_last_writes[j].end(); it++)
    if (thread_last_writes[l].find(*it) != thread_last_writes[l].end())
      return true;

  // This read what that wrote intersection
  for (std::set<expr2tc>::const_iterator it = thread_last_reads[j].begin();
       it != thread_last_reads[j].end(); it++)
    if (thread_last_writes[l].find(*it) != thread_last_writes[l].end())
      return true;

  // We wrote what that reads intersection
  for (std::set<expr2tc>::const_iterator it = thread_last_writes[j].begin();
       it != thread_last_writes[j].end(); it++)
    if (thread_last_reads[l].find(*it) != thread_last_reads[l].end())
      return true;

  // No check for read-read intersection, it doesn't affect anything
  return false;
}

void
execution_statet::calculate_mpor_constraints(void)
{
  // Primary bit of MPOR logic - to be executed at the end of a transition to
  // update dependency tracking and suchlike.

  // MPOR paper, page 12, create new dependency chain record for this time step.

  // 2D Vector of relations, T x T, i.e. threads on each axis:
  //    -1 signifies that no relation exists.
  //    0 that the thread hasn't run yet.
  //    1 that there is a dependency between these threads.
  //
  //  dependancy_chain contains the state from the previous transition taken;
  //  here we update it to reflect the latest transition, and make a decision
  //  about progress later.
  std::vector<std::vector<int> > new_dep_chain = dependancy_chain;

  // Start new dependency chain for this thread. Default to there being no
  // relation.
  for (unsigned int i = 0; i < new_dep_chain.size(); i++)
    new_dep_chain[active_thread][i] = -1;

  // This thread depends on this thread.
  new_dep_chain[active_thread][active_thread] = 1;

  // Mark un-run threads as continuing to be un-run. Otherwise, look for a
  // dependency chain from each thread to the run thread.
  for (unsigned int j = 0; j < new_dep_chain.size(); j++) {
    if (j == active_thread)
      continue;

    if (dependancy_chain[j][active_thread] == 0) {
      // This thread hasn't been run; continue not having been run.
      new_dep_chain[j][active_thread] = 0;
    } else {
      // This is where the beef is. If there is any other thread (including
      // the active thread) that we depend on, that depends on the active
      // thread, then record a dependency.
      // A direct dependency occurs when l = j, as DCjj always = 1, and DEPji
      // is true.
      int res = 0;

      for (unsigned int l = 0; l < new_dep_chain.size(); l++) {
        if (dependancy_chain[j][l] != 1)
          continue; // No dependency relation here

        // Now check for variable dependency.
        if (!check_mpor_dependancy(active_thread, l))
          continue;

        res = 1;
        break;
      }

      // Don't overwrite if no match
      if (res != 0)
        new_dep_chain[j][active_thread] = res;
    }
  }

  // For /all other relations/, just propagate the dependency it already has.
  // Achieved by initial duplication of dependancy_chain.

  // Voila, new dependency chain.

  // Calculate whether or not the transition we just took, in active_thread,
  // was in fact schedulable. We can't tell whether or not a transition is
  // allowed in advance because we don't know what it is. So instead, check
  // whether or not a transition /would/ have been allowed, once we've taken
  // it.
  bool can_run = true;
  for (unsigned int j = active_thread + 1; j < threads_state.size(); j++) {
    if (new_dep_chain[j][active_thread] != -1)
      // Either no higher threads have been run, or a dependency relation in
      // a higher thread justifies our out-of-order execution.
      continue;

    // Search for a dependency chain in a lower thread that links us back to
    // a higher thread, justifying this order.
    bool dep_exists = false;
    for (unsigned int l = 0; l < active_thread; l++) {
      if (dependancy_chain[j][l] == 1)
        dep_exists = true;
    }

    if (!dep_exists) {
      can_run = false;
      break;
    }
  }

  mpor_says_no = !can_run;

  dependancy_chain = new_dep_chain;
}

bool
execution_statet::has_cswitch_point_occured(void) const
{

  // Context switches can occur due to being forced, or by global state access

  if (cswitch_forced)
    return true;

  if (thread_last_reads[active_thread].size() != 0 ||
      thread_last_writes[active_thread].size() != 0)
    return true;

  return false;
}

bool
execution_statet::can_execution_continue(void) const
{

  if (threads_state[active_thread].thread_ended)
    return false;

  if (threads_state[active_thread].call_stack.empty())
    return false;

  return true;
}

crypto_hash
execution_statet::generate_hash(void) const
{

  auto l2 =
    std::dynamic_pointer_cast<state_hashing_level2t>(state_level2);
  assert(l2 != nullptr);

  crypto_hash state = l2->generate_l2_state_hash();
  std::string str = state.to_string();

  for (std::vector<goto_symex_statet>::const_iterator it = threads_state.begin();
       it != threads_state.end(); it++) {
    goto_programt::const_targett pc = it->source.pc;
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

crypto_hash
execution_statet::update_hash_for_assignment(const expr2tc &rhs)
{

  crypto_hash h;
  rhs->hash(h);
  h.fin();
  return h;
}

void
execution_statet::print_stack_traces(unsigned int indent) const
{
  std::vector<goto_symex_statet>::const_iterator it;
  std::string spaces = std::string("");
  unsigned int i;

  for (i = 0; i < indent; i++)
    spaces += " ";

  i = 0;
  for (it = threads_state.begin(); it != threads_state.end(); it++) {
    std::cout << spaces << "Thread " << i++ << ":" << std::endl;
    it->print_stack_trace(indent + 2);
    std::cout << std::endl;
  }

  return;
}

void
execution_statet::switch_to_monitor(void)
{

  if (threads_state[monitor_tid].thread_ended) {
    if (!mon_thread_warning) {
      std::cerr << "Switching to ended monitor; you need to increase its context or prefix bound" << std::endl;
      mon_thread_warning = true;
    }

    return;
  }

  assert(tid_is_set && "Must set monitor thread before switching to monitor\n");
  assert(!mon_from_tid &&"Switching to monitor without having switched away\n");

  monitor_from_tid = active_thread;
  mon_from_tid = true;

  if (monitor_tid != get_active_state_number()) {
    // Don't call switch_to_thread -- it'll execute the thread guard, which is
    // an extremely bad plan.
    last_active_thread = active_thread;
    active_thread = monitor_tid;
    cur_state = &threads_state[active_thread];
    cur_state->guard = threads_state[last_active_thread].guard;
  } else {
    assert(0 && "Switching to monitor thread from self\n");
  }
}

void
execution_statet::switch_away_from_monitor(void)
{

  // Occurs when we rerun the automata to discover whether or not the property
  // has been violated or not.
  if (threads_state[monitor_tid].thread_ended)
    return;

  assert(tid_is_set && "Must set monitor thread before switching from mon\n");
  assert(mon_from_tid && "Switching from monitor without switching to\n");

  assert(monitor_tid == active_thread &&
         "Must call switch_from_monitor from monitor thread\n");

  // Don't call switch_to_thread -- it'll execute the thread guard, which is
  // an extremely bad plan.
  last_active_thread = active_thread;
  active_thread = monitor_from_tid;
  cur_state = &threads_state[active_thread];

  cur_state->guard = threads_state[monitor_tid].guard;

  mon_from_tid = false;
}

void
execution_statet::kill_monitor_thread(void)
{
  assert(monitor_tid != active_thread &&
         "You cannot kill monitor thread _from_ the monitor thread\n");

  threads_state[monitor_tid].thread_ended = true;
  return;
}

static void replace_symbol_names(exprt &e, std::string prefix, std::map<std::string, std::string> &strings, std::set<std::string> &used_syms)
{

  if (e.id() ==  "symbol") {
    std::string sym = e.identifier().as_string();
    used_syms.insert(sym);
  } else {
    Forall_operands(it, e)
      replace_symbol_names(*it, prefix, strings, used_syms);
  }

  return;
}

void
execution_statet::init_property_monitors(void)
{
  std::map<std::string, std::string> strings;

  new_context.foreach_operand(
    [&strings] (const symbolt& s)
    {
      if (s.name.as_string().find("__ESBMC_property_") != std::string::npos) {
        // Munge back into the shape of an actual string
        std::string str = "";
        forall_operands(iter2, s.value) {
          char c = (char)strtol(iter2->value().as_string().c_str(), NULL, 2);
          if (c != 0)
            str += c;
          else
            break;
        }

        strings[s.name.as_string()] = str;
      }
    }
  );

  std::map<std::string, std::pair<std::set<std::string>, exprt> > monitors;
  std::map<std::string, std::string>::const_iterator str_it;
  for (str_it = strings.begin(); str_it != strings.end(); str_it++) {
    if (str_it->first.find("$type") == std::string::npos) {
      std::set<std::string> used_syms;
      exprt main_expr;
      std::string prop_name = str_it->first.substr(20, std::string::npos);

      namespacet ns(new_context);
      languagest languages(ns, MODE_C);

      std::string expr_str = strings["c::__ESBMC_property_" + prop_name];
      std::string dummy_str = "";

      languages.to_expr(expr_str, dummy_str, main_expr, message_handler);

      replace_symbol_names(main_expr, prop_name, strings, used_syms);

      monitors[prop_name] = std::pair<std::set<std::string>, exprt>
                                      (used_syms, main_expr);
    }
  }
}

execution_statet::ex_state_level2t::ex_state_level2t(
    execution_statet &ref)
  : renaming::level2t(),
    owner(&ref)
{
}

execution_statet::ex_state_level2t::~ex_state_level2t(void)
{
}

std::shared_ptr<renaming::level2t>
execution_statet::ex_state_level2t::clone(void) const
{

  return std::shared_ptr<ex_state_level2t>(new ex_state_level2t(*this));
}

void
execution_statet::ex_state_level2t::rename(expr2tc &lhs_sym, unsigned count)
{
  renaming::level2t::coveredinbees(lhs_sym, count, owner->node_id);
}

void
execution_statet::ex_state_level2t::rename(expr2tc &identifier)
{
  renaming::level2t::rename(identifier);
}

dfs_execution_statet::~dfs_execution_statet(void)
{

  // Delete target; or if we're encoding at runtime, pop a context.
  if (smt_during_symex)
    target->pop_ctx();
}

std::shared_ptr<execution_statet> dfs_execution_statet::clone(void) const
{
  std::shared_ptr<dfs_execution_statet> d =
    std::shared_ptr<dfs_execution_statet>(new dfs_execution_statet(*this));

  // Duplicate target equation; or if we're encoding at runtime, push a context.
  if (smt_during_symex) {
    d.get()->target = target;
    d.get()->target->push_ctx();
  } else {
    d.get()->target = target.get()->clone();
  }

  return d;
}

dfs_execution_statet::dfs_execution_statet(const dfs_execution_statet &ref)
  :  execution_statet(ref)
{
}

schedule_execution_statet::~schedule_execution_statet(void)
{
  // Don't delete equation. Schedule requires all this data.
}

std::shared_ptr<execution_statet> schedule_execution_statet::clone(void) const
{
  std::shared_ptr<schedule_execution_statet> s =
    std::shared_ptr<schedule_execution_statet>(new schedule_execution_statet(*this));

  // Don't duplicate target equation.
  s.get()->target = target;
  return s;
}

schedule_execution_statet::schedule_execution_statet(const schedule_execution_statet &ref)
  :  execution_statet(ref),
     ptotal_claims(ref.ptotal_claims),
     premaining_claims(ref.premaining_claims)
{
}

void
schedule_execution_statet::claim(const expr2tc &expr, const std::string &msg)
{
  unsigned int tmp_total, tmp_remaining;

  tmp_total = total_claims;
  tmp_remaining = remaining_claims;

  execution_statet::claim(expr, msg);

  tmp_total = total_claims - tmp_total;
  tmp_remaining = remaining_claims - tmp_remaining;

  *ptotal_claims += tmp_total;
  *premaining_claims += tmp_remaining;
  return;
}

execution_statet::state_hashing_level2t::state_hashing_level2t(
                                         execution_statet &ref)
    : ex_state_level2t(ref)
{
}

execution_statet::state_hashing_level2t::~state_hashing_level2t(void)
{
}

std::shared_ptr<renaming::level2t>
execution_statet::state_hashing_level2t::clone(void) const
{

  return std::shared_ptr<state_hashing_level2t>(new state_hashing_level2t(*this));
}

void
execution_statet::state_hashing_level2t::make_assignment(expr2tc &lhs_sym,
                                       const expr2tc &const_value,
                                       const expr2tc &assigned_value)
{
//  crypto_hash hash;

  renaming::level2t::make_assignment(lhs_sym, const_value, assigned_value);

  // If there's no body to the assignment, don't hash.
  if (!is_nil_expr(assigned_value)) {

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

  uint8_t *data = (uint8_t*)alloca(current_hashes.size() * CRYPTO_HASH_SIZE * sizeof(uint8_t));

  total = 0;
  for (current_state_hashest::const_iterator it = current_hashes.begin();
        it != current_hashes.end(); it++) {
    memcpy(&data[total * CRYPTO_HASH_SIZE], it->second.hash, CRYPTO_HASH_SIZE);
    total++;
  }

  crypto_hash c;
  c.ingest(data, total * CRYPTO_HASH_SIZE);
  c.fin();
  return c;
}
