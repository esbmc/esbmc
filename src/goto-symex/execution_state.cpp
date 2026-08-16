#include <goto-symex/execution_state.h>
#include <goto-symex/reachability_tree.h>
#include <goto-symex/symex_invariant.h>
#include <langapi/language_ui.h>
#include <langapi/languages.h>
#include <langapi/mode.h>
#include <sstream>
#include <string>
#include <util/base/breakpoint.h>
#include <util/lang/c_types.h>
#include <util/config/config.h>
#include <util/expr/expr_util.h>
#include <util/expr/type_byte_size.h>
#include <util/base/i2string.h>
#include <irep2/irep2.h>
#include <util/irep/migrate.h>
#include <util/irep/std_expr.h>
#include <vector>
#include <util/base/yaml_parser.h>

thread_local unsigned int execution_statet::node_count = 0;
thread_local unsigned int execution_statet::dynamic_counter = 0;

execution_statet::execution_statet(
  const goto_functionst &goto_functions,
  const namespacet &ns,
  reachability_treet *art,
  std::shared_ptr<symex_targett> _target,
  contextt &context,
  std::shared_ptr<ex_state_level2t> l2init,
  optionst &options)
  : goto_symext(ns, context, goto_functions, std::move(_target), options),
    state_level2(std::move(l2init)),
    global_value_set(ns)
{
  art1 = art;
  CS_number = 0;
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
  no_propagation = options.get_bool_option("no-propagation");

  goto_functionst::function_mapt::const_iterator it =
    goto_functions.function_map.find("__ESBMC_main");
  if (it == goto_functions.function_map.end())
  {
    log_error("main symbol not found; please set an entry point");
    abort();
  }

  const goto_programt *goto_program = &(it->second.body);

  // Initialize initial thread state
  goto_symex_statet state(*state_level2, global_value_set, ns, no_propagation);
  state.initialize(
    (*goto_program).instructions.begin(),
    (*goto_program).instructions.end(),
    goto_program,
    0);

  if (validate_witness)
  {
    const std::string &witness_path = options.get_option("witness");
    if (!witness_path.empty())
    {
      const auto &wps = yaml_parser::get_waypoints(witness_path);
      if (!wps.empty())
      {
        state.witness_segs.resize(wps.back().segment_idx + 1);
        for (const auto &wp : wps)
          state.witness_segs[wp.segment_idx].push_back(wp);
      }
    }
  }

  threads_state.push_back(state);
  preserved_paths.emplace_back();
  cur_state = &threads_state.front();
  cur_state->global_guard.make_true();
  cur_state->global_guard.add(get_guard_identifier());

  atomic_numbers.push_back(0);

  thread_start_data.emplace_back();

  // Initial mpor tracking.
  thread_last_reads.emplace_back();
  thread_last_writes.emplace_back();
  // One thread with one dependency relation.
  dependency_chain.emplace_back();
  dependency_chain.back().push_back(0);
  thread_last_transition.push_back(0);
  transition_ordinal = 0;
  mpor_says_no = false;

  cswitch_forced = false;
  active_thread = 0;
  last_active_thread = 0;
  node_count = 0;
  nondet_count = 0;

  thread_cswitch_threshold = (options.get_bool_option("ltl")) ? 3 : 2;
}

execution_statet::execution_statet(const execution_statet &ex)
  : goto_symext(ex),
    state_level2(
      std::dynamic_pointer_cast<ex_state_level2t>(ex.state_level2->clone())),
    global_value_set(ex.global_value_set)
{
  // The base slice is already copied by goto_symext(ex) in the
  // initialiser list; state_level2 / global_value_set are cloned there
  // too. copy_derived_from populates execution_statet's own fields
  // without touching the base, state_level2, threads_state, or
  // global_value_set.
  copy_derived_from(ex);

  // Regenerate threads state using the cloned level2/value_set so each
  // goto_symex_statet refers to the new instance, not the source's.
  threads_state.clear();
  for (const auto &t : ex.threads_state)
    threads_state.emplace_back(t, *state_level2, global_value_set);

  // Reassign which state is currently being worked on.
  cur_state = &threads_state[active_thread];
}

void execution_statet::copy_derived_from(const execution_statet &ex)
{
  // Only execution_statet's own scheduling fields are touched here. The
  // goto_symext base, state_level2, threads_state, and global_value_set
  // are deliberately omitted: the copy constructor handles them via its
  // initialiser list (base + state_level2 clone + global_value_set) and
  // body (threads_state rebuilt against the cloned level2).

  preserved_paths = ex.preserved_paths;
  atomic_numbers = ex.atomic_numbers;
  thread_start_data = ex.thread_start_data;
  last_active_thread = ex.last_active_thread;
  last_transition = ex.last_transition;
  active_thread = ex.active_thread;
  guard_execution = ex.guard_execution;
  nondet_count = ex.nondet_count;
  node_id = ex.node_id;
  interleaving_unviable = ex.interleaving_unviable;

  monitor_tid = ex.monitor_tid;
  tid_is_set = ex.tid_is_set;
  monitor_from_tid = ex.monitor_from_tid;
  mon_from_tid = ex.mon_from_tid;
  thread_cswitch_threshold = ex.thread_cswitch_threshold;
  symex_trace = ex.symex_trace;
  smt_during_symex = ex.smt_during_symex;
  smt_thread_guard = ex.smt_thread_guard;
  no_propagation = ex.no_propagation;

  CS_number = ex.CS_number;

  thread_last_reads = ex.thread_last_reads;
  thread_last_writes = ex.thread_last_writes;
  dependency_chain = ex.dependency_chain;
  thread_last_transition = ex.thread_last_transition;
  transition_ordinal = ex.transition_ordinal;
  mpor_says_no = ex.mpor_says_no;
  cswitch_forced = ex.cswitch_forced;

  state_level2->owner = this;
}

/* Mutex / condition-var / rwlock / barrier / spinlock types, looking through
 * arrays: `pthread_mutex_t m[N]` collects as the array symbol, whose type is
 * an array rather than the struct (#6480). */
static bool is_pthread_sync_type(type2tc t)
{
  while (is_array_type(t))
    t = to_array_type(t).subtype;
  if (is_nil_type(t))
    return false;
  if (is_struct_type(t))
  {
    const std::string &n = to_struct_type(t).name.as_string();
    return n.find("pthread_mutex_t") != std::string::npos ||
           n.find("pthread_cond_t") != std::string::npos ||
           n.find("pthread_rwlock_t") != std::string::npos ||
           n.find("pthread_barrier_t") != std::string::npos ||
           n.find("pthread_spinlock_t") != std::string::npos;
  }
  if (is_union_type(t))
  {
    const std::string &n = to_union_type(t).name.as_string();
    return n.find("pthread_mutex_t") != std::string::npos ||
           n.find("pthread_cond_t") != std::string::npos ||
           n.find("pthread_rwlock_t") != std::string::npos;
  }

  return false;
}

/* Build the MPOR key for element `elem` of a lock array. Both the pointer
 * path (which knows a byte offset) and a direct `m[i]` access (which knows an
 * element index) funnel through here so the two produce the same key. */
static expr2tc mpor_lock_array_key(const expr2tc &array, const BigInt &elem)
{
  const type2tc &subtype = to_array_type(array->type).subtype;
  return index2tc(subtype, array, constant_int2tc(index_type2(), elem));
}

/* Is `e` an array of pthread sync objects that we key per element? */
static bool is_lock_array(const expr2tc &e)
{
  return is_array_type(e->type) && is_pthread_sync_type(e->type);
}

/* Two MPOR keys conflict when they may name the same storage. A refined
 * lock-array key (index2t over the array symbol, see get_expr_globals) and the
 * whole-array key for that same symbol may name the same element, so they must
 * be treated as conflicting; two refined keys for distinct elements may not. */
static bool mpor_keys_may_alias(const expr2tc &a, const expr2tc &b)
{
  if (a == b)
    return true;
  if (is_index2t(a) == is_index2t(b))
    return false;
  const expr2tc &base_a = is_index2t(a) ? to_index2t(a).source_value : a;
  const expr2tc &base_b = is_index2t(b) ? to_index2t(b).source_value : b;
  return base_a == base_b;
}

static bool mpor_set_conflicts(const std::set<expr2tc> &s, const expr2tc &key)
{
  for (const expr2tc &e : s)
    if (mpor_keys_may_alias(e, key))
      return true;
  return false;
}

void execution_statet::symex_step(reachability_treet &art)
{
  statet &state = get_active_state();
  const goto_programt::instructiont &instruction = *state.source.pc;
  last_transition = transition_resultt();
  last_transition.thread_id = active_thread;

  merge_gotos();

  // If current state guard is false, it shouldn't perform further context
  // switch.
  if (
    !state.guard.is_false() || !is_cur_state_guard_false(state.guard.as_expr()))
    interleaving_unviable = false;
  else
    interleaving_unviable = true;

  // Don't convert if it's a inductive instruction and we are running the base
  // case or forward condition
  if (
    (base_case || forward_condition) && instruction.inductive_step_instruction)
  {
    // This assertion will prevent us of having weird side-effects (issue #538)
    // e.g. having inductive step instructions in a incremental strategy.
    // --termination also legitimately produces inductive-step instructions
    // (the post-main assert(false) marker inserted by goto_termination).
    assert(
      (k_induction || options.get_bool_option("termination")) &&
      "Inductive step instructions should be set only for k-induction "
      "or termination");
    cur_state->source.pc++;
    return;
  }

  if (options.get_bool_option("show-symex-value-sets"))
  {
    log_status("");
    state.value_set.dump();
  }

  if (symex_trace || options.get_bool_option("show-symex-value-sets"))
  {
    std::ostringstream oss;
    state.source.pc->output_instruction(ns, "", oss, false);
    log_result("{}", oss.str());
  }

  // We use this to break when we are about to run an instruction through symex
  if (break_insn != 0 && break_insn == instruction.location_number)
    breakpoint();

  switch (instruction.type)
  {
  case END_FUNCTION:
    if (instruction.function == "__ESBMC_main")
      end_thread();
    else if (
      (instruction.function == "c:@F@main" ||
       instruction.function == "c:@F@main#") &&
      !options.get_bool_option("memory-leak-check") &&
      !options.get_bool_option("termination"))
    {
      // check whether we reached the end of the main function and
      // whether we are not checking for memory leaks.
      // We should end the main thread to avoid exploring further interleavings
      // TODO: once we support at_exit, we should check this code
      // TODO: we should support verifying memory leaks in multi-threaded C
      // programs.
      //
      // Returning from main is a call to exit(): the process terminates and
      // every other thread is torn down (C11 5.1.2.2.3), so a thread still
      // waiting on a lock main held is not deadlocked. This END_FUNCTION is
      // reached only on a real return -- pthread_exit() ends in
      // __ESBMC_assume(0) -- so a program whose threads outlive main via
      // pthread_exit keeps being explored under --deadlock-check (#6479).
      //
      // Skipped under --termination: the strategy inserts an
      // `assert(false)` after the main() call in __ESBMC_main to
      // detect terminating paths. The assume(false) here would kill
      // the path before reaching that marker; the assertion would
      // never become a VCC.
      assume(gen_false_expr());
      end_thread();
      interleaving_unviable = true;
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
    break;
  case RETURN:
    if (
      !state.guard.is_false() ||
      !is_cur_state_guard_false(state.guard.as_expr()))
    {
      expr2tc thecode = instruction.code, assign;
      if (make_return_assignment(assign, thecode))
      {
        auto saved_source = cur_state->source;
        cur_state->source = cur_state->top().calling_location;
        goto_symext::symex_assign(assign);
        cur_state->source = saved_source;
      }
      // symex_return falsifies the path guard, and analyze_assign ignores a
      // false guard, so a shared write performed by the return assignment
      // offers no context-switch point unless analyzed first. The guard a
      // switch here must chain is likewise the one from before the return
      // (issue #6558).
      analyze_assign(assign);
      last_transition.parent_guard = threads_state[active_thread].guard;
      symex_return(thecode);
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
  const guard2tc &guard)
{
  goto_symext::symex_assign(code, hidden, guard);

  if (threads_state.size() >= thread_cswitch_threshold)
    analyze_assign(code);
}

void execution_statet::claim(const expr2tc &expr, const std::string &msg)
{
  goto_symext::claim(expr, msg);

  if (threads_state.size() >= thread_cswitch_threshold)
    analyze_read(expr);
}

void execution_statet::symex_goto(const expr2tc &old_guard)
{
  last_transition.parent_guard = threads_state[active_thread].guard;

  // Analyze the guard's shared reads before executing the goto: a
  // constant-true guard kills the fall-through path guard, and analyzing
  // afterwards would then skip the read entirely, losing the context-switch
  // point at the branch (issue #6558).
  if (threads_state.size() >= thread_cswitch_threshold)
    analyze_read(old_guard);

  goto_symext::symex_goto(old_guard);
}

void execution_statet::record_parked_path(
  goto_programt::const_targett target,
  statet::merge_state_listt::iterator parked)
{
  last_transition.parked = parked_patht{target, parked};
}

void execution_statet::assume(const expr2tc &assumption)
{
  goto_symext::assume(assumption);

  if (threads_state.size() >= thread_cswitch_threshold)
    analyze_read(assumption);
}

void execution_statet::symex_printf(const expr2tc &lhs, expr2tc &code)
{
  // A shared global passed to printf is read here and nowhere else, so without
  // this the read is missing from thread_last_reads and every reduction that
  // consults it treats the transition as independent of a writer (issue #6831).
  if (threads_state.size() >= thread_cswitch_threshold)
    analyze_read(code);

  goto_symext::symex_printf(lhs, code);
}

void execution_statet::note_bounded_loop_truncation()
{
  goto_symext::note_bounded_loop_truncation();
  art1->mark_search_truncated();
}

void execution_statet::reset_dynamic_counter()
{
  dynamic_counter = 0;
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

unsigned int execution_statet::get_active_atomic_number() const
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
    symbol_renaming_level::level1,
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

bool execution_statet::check_if_ileaves_blocked()
{
  if (art1->get_CS_bound() != -1 && CS_number >= art1->get_CS_bound())
  {
    // Only count this as truncation if a switch was actually available;
    // otherwise every terminal state would look truncated.
    for (unsigned int i = 0; i < threads_state.size(); ++i)
      if (
        i != active_thread && !threads_state[i].thread_ended &&
        !threads_state[i].call_stack.empty())
      {
        art1->cs_bound_pruned = true;
        ++art1->reduction_stats.pruned_by_cs_bound;
        art1->mark_search_truncated();
        break;
      }
    return true;
  }

  if (get_active_atomic_number() > 0)
    return true;

  // A directed monitor step runs from switch_to_monitor to its paired switch
  // away. The caller's ATOMIC_BEGIN does not cover it -- atomic counts are
  // per-thread and the monitor's own is zero -- so interleaving here would
  // abandon the monitor mid-step and strand the bookkeeping (#6585). The
  // instrumented sites in property_monitors.cpp are all atomic anyway; the one
  // exception is libltl2ba's own switch in ltl2ba_start_monitor, which runs
  // before any user thread exists.
  if (mon_from_tid)
    return true;

  if (art1->directed_interleavings)
    // Don't generate interleavings automatically - instead, the user will
    // inserts intrinsics identifying where they want interleavings to occur,
    // and to what thread.
    return true;

  // Don't generate further interleavings since the __ESBMC_main thread has
  // ended. Whether it has is a property of *this* state, not of the search:
  // the reachability tree's own flag is set once and never cleared on
  // backtracking, so a single branch running main to completion used to
  // disable interleaving for every branch explored afterwards -- which is
  // where the racy schedules live (#4584).
  if (
    !threads_state.empty() && threads_state[0].thread_ended &&
    !options.get_bool_option("deadlock-check") &&
    !options.get_bool_option("data-races-check"))
    return true;

  // The monitor never counts: it is stepped by directed switches, so it does
  // not make an otherwise single-threaded program worth interleaving (#6585).
  if (threads_state.size() < 2u + (tid_is_set ? 1u : 0u))
    return true;

  return false;
}

bool execution_statet::all_threads_terminal() const
{
  for (const auto &ts : threads_state)
    if (!ts.thread_ended && !ts.call_stack.empty())
      return false;
  return true;
}

void execution_statet::end_thread()
{
  get_active_state().thread_ended = true;
  // If ending in an atomic block, the switcher fails to switch to another
  // live thread (because it's trying to be atomic). So, disable atomic blocks
  // when the thread ends.
  atomic_numbers[active_thread] = 0;

  // A monitor that runs out of prefix bound ends mid-step, so its paired
  // switch away never clears the in-flight flag. Left set it would block
  // every later interleaving, the same way a stale atomic count would.
  if (tid_is_set && active_thread == monitor_tid)
    mon_from_tid = false;
}

void execution_statet::update_after_switch_point()
{
  execute_guard();

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
  if (last_active_thread != active_thread)
  {
    preserve_last_paths(last_transition);
    cull_all_paths();
    restore_last_paths();
  }
}

void execution_statet::preserve_last_paths(const transition_resultt &transition)
{
  // If the thread terminated, there are no paths to preserve: this is the final
  // switching away.
  if (threads_state[last_active_thread].thread_ended)
    return;

  // Examine the current execution state and the explicit result of the last
  // transition, deciding which paths are going to be preserved after this
  // context switch. The current instruction and guard are guaranteed (unless
  // the guard is false), and any branch path generated by the transition has
  // already been recorded by the GOTO symex step.

  auto &pp = preserved_paths[last_active_thread];
  auto &ls = threads_state[last_active_thread];
  assert(pp.size() == 0 && "Unmerged preserved paths in ex_state");
  assert(
    transition.thread_id == last_active_thread &&
    "Preserving paths for the wrong transition thread");

  // Add the current path to the set of paths to be preserved. Don't do this
  // if the current guard is false, though.
  if (!ls.guard.is_false() || !is_cur_state_guard_false(ls.guard.as_expr()))
    pp.push_back(std::make_pair(ls.source.pc, merge_statet(ls)));

  if (transition.parked)
  {
    // The GOTO or RETURN that produced this transition pushed a merge_statet
    // onto ls.top().merge_state_map[transition.parked->target]. We captured
    // an iterator to it at the time, so no further scan or guard matching
    // is needed.
    //
    // Sanity: the map entry for the recorded target must still exist.
    // See parked_patht::snapshot docs for why the iterator can survive
    // the clone but stays load-bearing: this assertion does NOT catch
    // a dangling iterator, only the easier case where the list entry
    // itself disappeared.
    assert(
      ls.top().merge_state_map.count(transition.parked->target) &&
      "parked path target missing from merge_state_map");
    pp.emplace_back(
      transition.parked->target, merge_statet(*transition.parked->snapshot));
  }

  // We must have picked up at least one path to merge
  if (pp.size() == 0)
  {
    // Even better: if the guard is now false, and we're context switching away,
    // then something like assume(0) occurred: no paths can continue in this
    // thread from this point. And anything we context switch to will get a
    // false guard too and thus expire.
    // Ideally at this point we would bail and return our formula to the RT
    // class, but that code is way too fragile. Instead, continue with an ended
    // thread that infects all other threads with it's false guard until we
    // complete.
    // We can't tell the assume(0) case apart from an internal logic bug that
    // also produced an empty preserved-paths list. Log so traces are
    // greppable; the normal path is a legitimate assume(0).
    log_debug(
      "symex",
      "preserve_last_paths: no paths preserved for thread {}, ending it "
      "(usually assume(0); investigate if you don't expect one here)",
      last_active_thread);
    threads_state[last_active_thread].thread_ended = true;
    atomic_numbers[last_active_thread] = 0;
  }
}

void execution_statet::cull_all_paths()
{
  // A false guard means this thread's live continuations are its parked
  // merge states. If the switch-out preserved nothing, keep them (GitHub
  // issue #608). When paths were preserved, fall through and clear:
  // restore_last_paths re-parks them against the present level2 values, and
  // the stale twins left behind here would otherwise resurrect pre-switch
  // values at the merge, losing every write other threads made since the
  // branch (issue #6558).
  if (
    (is_false(cur_state->guard.as_expr()) ||
     is_cur_state_guard_false(cur_state->guard.as_expr())) &&
    preserved_paths[active_thread].empty())
    return;

  // Walk through _all_ symbolic paths in the program and wipe them out.
  // Current path is easy: set the guard to false. phi_function will overwrite
  // any different valuation left in the l2 map.
  cur_state->guard.make_false();

  // This completely removes all symbolic paths that were going to be merged
  // back in at some point in the future.
  for (auto &frame : cur_state->call_stack)
  {
    frame.merge_state_map.clear();
  }
}

void execution_statet::restore_last_paths()
{
  // For each preserved path: create a fresh new merge_statet with data values
  // created from the present values of l2-renaming and value set, as we
  // (presumably) switch back in from a different thread. Then schedule the
  // states to be merged in at their original locations.
  // Given that we're discarding a lot of data here this could all be more
  // efficient, but it's what we've got.

  auto &list = preserved_paths[active_thread];
  for (auto const &p : list)
  {
    const auto &loc = p.first;
    const auto &gs = p.second;

    // This is an experimental option to stop the verification process
    // if we have more than one state at a given location to merge.
    if (
      options.get_bool_option("no-goto-merge") &&
      cur_state->top().merge_state_map[loc].size() != 0)
    {
      log_error(
        "There are goto statements that shouldn't be merged at this point");
      abort();
    }
    // Create a fresh new merge_statet to be merged in at the target insn.
    // Capture the inserted element directly: if the list already has older
    // entries at this location (a different preserved path joining at the
    // same instruction), reading .begin() would alias the wrong entry and
    // corrupt it via the writes below.
    auto &new_gs =
      cur_state->top().merge_state_map[loc].emplace_back(*cur_state);

    // Proceed to fill new_gs with old data. Ideally this would be a method...
    new_gs.num_instructions = gs.num_instructions;
    new_gs.guard = gs.guard;
    assert(new_gs.thread_id == gs.thread_id);
  }

  list.clear();
}

bool execution_statet::is_cur_state_guard_false(const expr2tc &guard)
{
  // So, can the assumption actually be true? If enabled, ask the solver.
  if (smt_thread_guard)
  {
    expr2tc parent_guard = guard;

    runtime_encoded_equationt *rte =
      dynamic_cast<runtime_encoded_equationt *>(target.get());

    expr2tc the_question = equality2tc(gen_true_expr(), parent_guard);

    try
    {
      tvt res = rte->ask_solver_question(the_question);
      if (res.is_false())
        return true;
    }
    catch (runtime_encoded_equationt::dual_unsat_exception &e)
    {
      // Basically, this means our _assumptions_ here are false as well, so
      // neither true or false guards are possible. Consider this as meaning
      // that the guard is false.
      return true;
    }
  }

  return is_false(guard);
}

void execution_statet::execute_guard()
{
  node_id = node_count++;
  expr2tc guard_expr = get_guard_identifier();
  expr2tc parent_guard;

  // Preserve the existing false-guard handling for context switches taken
  // immediately after a branch, but keep the branch parent state explicit.
  if (last_transition.parent_guard && last_transition.parent_guard->is_false())
  {
    guard2tc tmp = *last_transition.parent_guard;
    tmp |= threads_state[last_active_thread].guard;
    parent_guard = tmp.as_expr();
  }
  else if (last_transition.parked && last_transition.parent_guard)
  {
    // Switching right after a step that parked the thread's continuation:
    // chain the guard from before it was parked, since the present one is
    // false. Using it poisons the suffix with assume(false) -- when a
    // constant-true goto killed the fall-through, and at every return, where
    // symex_return always falsifies the guard (issue #6558).
    parent_guard = last_transition.parent_guard->as_expr();
  }
  else
  {
    parent_guard = threads_state[last_active_thread].guard.as_expr();
  }

  // If we simplified the global guard expr to false, write that to thread
  // guards, not the symbolic guard name. This is the only way to bail out of
  // evaluating a particular interleaving early right now.
  if (is_false(parent_guard) || is_cur_state_guard_false(parent_guard))
  {
    // A context switch happens, add last thread state guard to assumption.
    if (active_thread != last_active_thread)
    {
      target->assumption(
        guard2tc().as_expr(),
        parent_guard,
        get_active_state().source,
        first_loop);
    }
    cur_state->guard.make_false();
    return;
  }

  // Rename value, allows its use in other renamed exprs
  state_level2->make_assignment(guard_expr, expr2tc(), expr2tc());

  // Truth of this guard implies the parent is true.
  state_level2->rename(parent_guard);
  do_simplify(parent_guard);

  if (active_thread != last_active_thread)
    target->assumption(
      guard2tc().as_expr(),
      parent_guard,
      get_active_state().source,
      first_loop);
}

unsigned int execution_statet::add_thread(const goto_programt *prog)
{
  goto_symex_statet new_state(
    *state_level2, global_value_set, ns, no_propagation);
  new_state.initialize(
    prog->instructions.begin(),
    prog->instructions.end(),
    prog,
    threads_state.size());

  unsigned int thread_nr = threads_state.size();
  new_state.source.thread_nr = thread_nr;
  new_state.guard = cur_state->guard;
  new_state.global_guard.make_true();
  new_state.global_guard.add(get_guard_identifier());
  threads_state.push_back(new_state);
  if (threads_state.size() > art1->reduction_stats.peak_threads)
    art1->reduction_stats.peak_threads = threads_state.size();
  preserved_paths.emplace_back();
  atomic_numbers.push_back(0);

  thread_start_data.emplace_back();

  // We invalidated all threads_state refs, so reset cur_state ptr.
  cur_state = &threads_state[active_thread];

  // Update MPOR tracking data with newly initialized thread
  thread_last_reads.emplace_back();
  thread_last_writes.emplace_back();
  // Unfortunately as each thread has a dependency relation with every other
  // thread we have to do a lot of work to initialize a new one. And initially
  // all relations are '0', no transitions yet.
  for (auto &it : dependency_chain)
  {
    it.push_back(0);
  }
  // And the new threads dependencies,
  dependency_chain.emplace_back();
  for (unsigned int i = 0; i < dependency_chain.size(); i++)
    dependency_chain.back().push_back(0);
  thread_last_transition.push_back(0);

  // While we've recorded the new thread as starting in the designated program,
  // it might not run immediately, thus must have it's path preserved:
  preserved_paths[thread_nr].push_back(std::make_pair(
    prog->instructions.begin(), merge_statet(threads_state[thread_nr])));

  return threads_state.size() - 1; // thread ID, zero based
}

void execution_statet::analyze_assign(const expr2tc &code)
{
  if (is_nil_expr(code) || get_active_state().guard.is_false())
    return;

  std::set<expr2tc> global_reads, global_writes;
  const code_assign2t &assign = to_code_assign2t(code);
  get_expr_globals(ns, assign.target, global_writes, access_kindt::WRITE);
  get_expr_globals(ns, assign.source, global_reads, access_kindt::READ);

  if (global_reads.size() > 0 || global_writes.size() > 0)
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
  if (is_nil_expr(code) || get_active_state().guard.is_false())
    return;

  std::set<expr2tc> global_reads;
  get_expr_globals(ns, code, global_reads, access_kindt::READ);

  if (global_reads.size() > 0)
  {
    // Record read data
    thread_last_reads[active_thread].insert(
      global_reads.begin(), global_reads.end());
  }
}

void execution_statet::analyze_args(const expr2tc &expr)
{
  if (threads_state.size() >= thread_cswitch_threshold)
    analyze_read(expr);
}

expr2tc execution_statet::resolve_pointer_target(
  const namespacet &ns,
  const expr2tc &ptr,
  bool &to_global)
{
  expr2tc tmp = ptr;
  /* Rename it so that it can be dereferenced in current state */
  cur_state->rename(tmp);
  /* Both call sites check the operand is a pointer, so this is a precondition
   * rather than a case to handle: a guard here would be dead instrumentation.
   */
  SYMEX_INVARIANT(
    is_pointer_type(tmp->type), "resolving a non-pointer operand");

  /* Collect all the objects pointed to by the pointer */
  expr2tc deref = dereference2tc(to_pointer_type(tmp->type).subtype, tmp);
  value_setst::valuest dest;
  cur_state->value_set.get_reference_set(deref, dest);

  expr2tc found;
  for (const auto &obj : dest)
  {
    if (
      !is_object_descriptor2t(obj) ||
      !is_symbol2t(to_object_descriptor2t(obj).object))
      continue;

    const std::string &n =
      to_symbol2t(to_object_descriptor2t(obj).object).thename.as_string();
    const symbolt *s = ns.lookup(n);
    if (!s)
      continue;
    if (is_esbmc_internal_symbol(n))
      continue;

    to_global = s->static_lifetime || s->get_type().is_dynamic_set();
    found = to_object_descriptor2t(obj).object;
    /* Distinguish the elements of a lock array. Both `&m[0]` and `&m[1]`
     * resolve to the base symbol `m`, which makes MPOR treat every lock in the
     * array as one object, so two threads holding different locks never come
     * out independent -- 6.8x the interleavings of the same program written
     * with scalar mutexes (#6480). Refine only a constant offset on a lock
     * array; an unknown offset keeps the whole-array key, and
     * mpor_keys_may_alias pairs the refined and unrefined forms. */
    const expr2tc &off = to_object_descriptor2t(obj).offset;
    if (is_constant_int2t(off) && is_lock_array(found))
    {
      /* The descriptor carries a byte offset; the key is an element index so
       * that it matches the one a direct `m[i]` access builds. */
      BigInt esize = type_byte_size(to_array_type(found->type).subtype, &ns);
      if (esize > 0)
        found =
          mpor_lock_array_key(found, to_constant_int2t(off).value / esize);
    }
    /* Stop when the global symbol is found */
    if (to_global)
      break;
  }
  return found;
}

void execution_statet::record_access_key(
  const expr2tc &key,
  std::set<expr2tc> &globals_list,
  access_kindt kind)
{
  // Read-only-global filter: a READ of a global that is never written
  // anywhere in the program cannot participate in a data race, so it
  // must not trigger a cswitch. WRITE accesses are never filtered --
  // a write on its own establishes the "may be written" fact for the
  // other side of any future read/write conflict.
  if (art1->readonly_global_opt && kind == access_kindt::READ)
  {
    expr2tc orig = key;
    get_active_state().get_original_name(orig);
    if (is_symbol2t(orig))
    {
      const irep_idt &resolved_name = to_symbol2t(orig).thename;
      if (!art1->may_be_written(resolved_name))
        return;
    }
  }

  expr2tc p = key;
  std::list<unsigned int> threadId_list;
  auto it_find = art1->vars_map.find(p);

  // the expression was accessed in another interleaving
  if (it_find != art1->vars_map.end())
  {
    threadId_list = it_find->second;
    if (
      std::find(threadId_list.begin(), threadId_list.end(), active_thread) ==
      threadId_list.end())
    {
      it_find->second.push_back(active_thread);
    }

    std::list<unsigned int>::iterator it_list;
    for (it_list = threadId_list.begin(); it_list != threadId_list.end();
         ++it_list)
    {
      // find if some thread access the same expression
      if (*it_list != active_thread)
      {
        globals_list.insert(p);
        art1->is_global.insert(p);
      }
      // expression was not accessed by other thread
      else
      {
        auto its_global = art1->is_global.find(p);
        // expression was defined as global in another interleaving
        if (its_global != art1->is_global.end())
          globals_list.insert(p);
      }
    }
    // first access of expression
  }
  else
  {
    auto its_global = art1->is_global.find(p);
    if (its_global != art1->is_global.end())
      globals_list.insert(p);
    else
    {
      threadId_list.push_back(active_thread);
      art1->vars_map.insert(
        std::pair<expr2tc, std::list<unsigned int>>(p, threadId_list));
      globals_list.insert(p);
      art1->is_global.insert(p);
    }
  }
}

/* A dereference through a pointer that is not spelled as a bare symbol -- held
 * in a struct member, array element or union -- never reaches the symbol
 * resolution in get_expr_globals, which is gated on `is_symbol2t`. Its operand
 * walk would then record the *aggregate*, so a thread reaching the same target
 * directly keys on something else and MPOR calls the two transitions
 * independent (R29, the aggregate-held counterpart of #6539). Record the target
 * as well; the walk still runs, and recording both keys only makes MPOR more
 * conservative. */
void execution_statet::record_aggregate_held_target(
  const namespacet &ns,
  const expr2tc &expr,
  std::set<expr2tc> &globals_list,
  access_kindt kind)
{
  if (!is_dereference2t(expr))
    return;

  const expr2tc &ptr = to_dereference2t(expr).value;
  if (is_symbol2t(ptr) || !is_pointer_type(ptr->type))
    return;

  bool to_global = false;
  expr2tc target = resolve_pointer_target(ns, ptr, to_global);
  if (is_nil_expr(target) || !to_global)
    return;

  cur_state->top().level1.rename(target);
  record_access_key(target, globals_list, kind);
}

void execution_statet::get_expr_globals(
  const namespacet &ns,
  const expr2tc &expr,
  std::set<expr2tc> &globals_list,
  access_kindt kind)
{
  if (options.get_bool_option("data-races-check-only"))
    return;

  if (is_nil_expr(expr))
    return;

  if (
    is_address_of2t(expr) || is_valid_object2t(expr) || is_dynamic_size2t(expr))
  {
    return;
  }
  if (is_symbol2t(expr))
  {
    expr2tc newexpr = expr;
    get_active_state().get_original_name(newexpr);
    const std::string &name = to_symbol2t(newexpr).thename.as_string();

    if (
      name == "goto_symex::guard!" +
                i2string(get_active_state().top().level1.thread_id))
      return;

    const symbolt *symbol = ns.lookup(name);
    if (!symbol)
      return;

    // Resolve pointer parameters/locals BEFORE applying the internal-name
    // filter. The pointer variable itself may live in pthread_lib (e.g. the
    // `mutex` parameter of pthread_mutex_lock) and so match the filter, yet
    // still point to a user global whose access must be recorded as an MPOR
    // dependency.
    expr2tc p = expr;
    bool point_to_global = false;
    std::vector<expr2tc> deeper_targets;
    if (symbol->get_type().is_pointer() && symbol->name != "invalid_object")
    {
      p = resolve_pointer_target(ns, expr, point_to_global);
      if (is_nil_expr(p))
        p = expr;

      // Follow the rest of the chain. Resolving a single level records the
      // *intermediate* pointer for an access spelled `*(*pp)`, so a thread
      // reaching the same object directly keys on something else and MPOR
      // calls the two transitions independent (#6539). Every object along the
      // way is recorded, not just the last: stopping at the first shared one
      // would not reach the target, and skipping to the target would lose a
      // shared intermediate. The visited set bounds a pointer cycle.
      std::set<std::string> seen;
      expr2tc cursor = p;
      while (is_symbol2t(cursor) && is_pointer_type(cursor->type) &&
             seen.insert(to_symbol2t(cursor).thename.as_string()).second)
      {
        bool deeper_is_global = false;
        expr2tc next = resolve_pointer_target(ns, cursor, deeper_is_global);
        if (is_nil_expr(next))
          break;
        if (deeper_is_global)
        {
          deeper_targets.push_back(next);
          point_to_global = true;
        }
        cursor = next;
      }
    }

    // Drop the pointer symbol itself if it is an internal pthread_lib name,
    // unless we've resolved it to a user global above.
    if (is_esbmc_internal_symbol(name) && !point_to_global)
      return;

    // Rename to level1 to avoid shared varible mismatch in mpor.
    cur_state->top().level1.rename(p);
    // Python module-level globals carry static_lifetime=false (their
    // symbol.value field doubles as a const-prop snapshot in the Python
    // frontend), but the frontend marks them with file_local=false to
    // signal "shared module state". Recognise them here so symex inserts
    // interleaving points at their reads/writes, matching the
    // race-eligible bypass added to rw_set.cpp.
    //
    // Necessary but not sufficient on its own: the scheduler-side DFS /
    // MPOR limiters tracked in #4584 also need to be loosened before
    // assertion-based race tests like increment_race flip to FAILED.
    const bool python_global = symbol->mode == "Python" && !symbol->file_local;
    if (
      symbol->static_lifetime || symbol->get_type().is_dynamic_set() ||
      point_to_global || python_global)
      record_access_key(p, globals_list, kind);

    // Objects further along the chain are shared on their own merit, so they
    // are recorded whether or not the pointer that reached them qualifies:
    // `int *lp = &g; int **lpp = &lp;` has no shared symbol until `g` (#6539).
    for (expr2tc deeper : deeper_targets)
    {
      cur_state->top().level1.rename(deeper);
      record_access_key(deeper, globals_list, kind);
    }
    return;
  }

  record_aggregate_held_target(ns, expr, globals_list, kind);

  /* A direct `m[i]` on a lock array: record the element. Falling through to
   * the operand walk below would reach the bare array symbol and record the
   * whole array, which re-conflates the elements the pointer path above took
   * care to separate (#6480). */
  if (is_index2t(expr))
  {
    const index2t &idx = to_index2t(expr);
    if (is_symbol2t(idx.source_value) && is_lock_array(idx.source_value))
    {
      expr2tc src = idx.source_value;
      get_active_state().get_original_name(src);
      const symbolt *s = ns.lookup(to_symbol2t(src).thename);
      expr2tc i = idx.index;
      cur_state->rename(i);
      simplify(i);
      if (
        s && (s->static_lifetime || s->get_type().is_dynamic_set()) &&
        is_constant_int2t(i))
      {
        src = idx.source_value;
        cur_state->top().level1.rename(src);
        globals_list.insert(
          mpor_lock_array_key(src, to_constant_int2t(i).value));
        return;
      }
    }
  }

  expr->foreach_operand([this, &globals_list, &ns, kind](const expr2tc &e) {
    get_expr_globals(ns, e, globals_list, kind);
  });
}

// Rules given on page 13 of MPOR paper, although they don't appear to
// distinguish which thread is which correctly. Essentially, check that the
// write(s) of the previous transition (l) don't intersect with this
// transitions (j) reads or writes; and that the previous transitions reads
// don't intersect with this transitions write(s).
static bool mpor_transitions_conflict(
  const std::set<expr2tc> &reads_j,
  const std::set<expr2tc> &writes_j,
  const std::set<expr2tc> &reads_l,
  const std::set<expr2tc> &writes_l)
{
  // Double write intersection
  for (const expr2tc &it : writes_j)
    if (mpor_set_conflicts(writes_l, it))
      return true;

  // This read what that wrote intersection
  for (const expr2tc &it : reads_j)
    if (mpor_set_conflicts(writes_l, it))
      return true;

  // We wrote what that reads intersection
  for (const expr2tc &it : writes_j)
    if (mpor_set_conflicts(reads_l, it))
      return true;

  // No check for read-read intersection, it doesn't affect anything
  return false;
}

bool execution_statet::check_mpor_dependency(
  unsigned int j,
  const transition_footprintt &fp) const
{
  assert(j < threads_state.size());
  return mpor_transitions_conflict(
    thread_last_reads[j], thread_last_writes[j], fp.reads, fp.writes);
}

bool execution_statet::check_mpor_dependency(unsigned int j, unsigned int l)
  const
{
  assert(j < threads_state.size());
  assert(l < threads_state.size());
  return mpor_transitions_conflict(
    thread_last_reads[j],
    thread_last_writes[j],
    thread_last_reads[l],
    thread_last_writes[l]);
}

/// The inductive step of A6.4: every 1 in the chain points forward in run
/// order. Only the active thread's row and column change meaning when it takes
/// a transition -- every other entry keeps both its endpoints -- so checking
/// those two is checking the step.
static void check_chain_points_forward(
  const std::vector<std::vector<int>> &chain,
  const std::vector<unsigned int> &last_transition,
  unsigned int active_thread,
  unsigned int j)
{
  // The column covers the res == 0 path, which keeps a 1 recorded against an
  // older transition of the active thread; the row covers the active-row reset,
  // without which the chain would claim a dependency leaving the newest
  // transition in the run.
  SYMEX_INVARIANT(
    chain[j][active_thread] != 1 ||
      last_transition[j] < last_transition[active_thread],
    "MPOR dependency chain runs backwards in time");
  SYMEX_INVARIANT(
    chain[active_thread][j] != 1 ||
      last_transition[active_thread] < last_transition[j],
    "MPOR dependency chain leaves the newest transition");
}

void execution_statet::calculate_mpor_constraints()
{
  // Primary bit of MPOR logic - to be executed at the end of a transition to
  // update dependency tracking and such like.

  // MPOR paper, page 12, create new dependency chain record for this time step.

  // 2D Vector of relations, T x T, i.e. threads on each axis:
  //    -1 signifies that no relation exists.
  //    0 that the thread hasn't run yet.
  //    1 that there is a dependency between these threads.
  //
  //  dependency_chain contains the state from the previous transition taken;
  //  here we update it to reflect the latest transition, and make a decision
  //  about progress later.
  std::vector<std::vector<int>> new_dep_chain = dependency_chain;

  // The transition just taken is the newest in the run.
  thread_last_transition[active_thread] = ++transition_ordinal;

  // Start new dependency chain for this thread. Default to there being no
  // relation.
  for (unsigned int i = 0; i < new_dep_chain.size(); i++)
    new_dep_chain[active_thread][i] = -1;

  // This thread depends on this thread.
  new_dep_chain[active_thread][active_thread] = 1;

  // Mark un-run threads as continuing to be un-run. Otherwise, look for a
  // dependency chain from each thread to the run thread.
  for (unsigned int j = 0; j < new_dep_chain.size(); j++)
  {
    if (j == active_thread)
      continue;

    // The diagonal is what the un-run test below reads, and it is only a
    // "has this thread run" flag because nothing else ever writes it.
    SYMEX_INVARIANT(
      (dependency_chain[j][j] == 1) == (thread_last_transition[j] != 0),
      "MPOR chain diagonal disagrees with whether the thread has run");

    // MPOR's rule is DCjj(k) == 0 -- "Tj has not run" -- not DCji(k) == 0. The
    // two differ for a thread created after Tj last ran: its column is padded
    // with 0, so testing it would skip the dependency below and drop the chain
    // onto the new thread's first transition (A6.4).
    if (dependency_chain[j][j] == 0)
    {
      // This thread hasn't been run; continue not having been run.
      new_dep_chain[j][active_thread] = 0;
    }
    else
    {
      // This is where the beef is. If there is any other thread (including
      // the active thread) that we depend on, that depends on the active
      // thread, then record a dependency.
      // A direct dependency occurs when l = j, as DCjj always = 1, and DEPji
      // is true.
      int res = 0;

      for (unsigned int l = 0; l < new_dep_chain.size(); l++)
      {
        if (dependency_chain[j][l] != 1)
          continue; // No dependency relation here

        // Now check for variable dependency.
        if (!check_mpor_dependency(active_thread, l))
          continue;

        res = 1;
        break;
      }

      // Don't overwrite if no match
      if (res != 0)
        new_dep_chain[j][active_thread] = res;
    }

    check_chain_points_forward(
      new_dep_chain, thread_last_transition, active_thread, j);
  }

  // For /all other relations/, just propagate the dependency it already has.
  // Achieved by initial duplication of dependency_chain.

  // Voila, new dependency chain.

  // Calculate whether or not the transition we just took, in active_thread,
  // was in fact schedulable. We can't tell whether or not a transition is
  // allowed in advance because we don't know what it is. So instead, check
  // whether or not a transition /would/ have been allowed, once we've taken
  // it.
  bool can_run = true;
  for (unsigned int j = active_thread + 1; j < threads_state.size(); j++)
  {
    if (new_dep_chain[j][active_thread] != -1)
      // Either no higher threads have been run, or a dependency relation in
      // a higher thread justifies our out-of-order execution.
      continue;

    // Search for a dependency chain in a lower thread that links us back to
    // a higher thread, justifying this order.
    bool dep_exists = false;
    for (unsigned int l = 0; l < active_thread; l++)
    {
      if (dependency_chain[j][l] == 1)
        dep_exists = true;
    }

    if (!dep_exists)
    {
      can_run = false;
      break;
    }
  }

  mpor_says_no = !can_run;

  dependency_chain = new_dep_chain;
}

bool execution_statet::has_cswitch_point_occured() const
{
  // Context switches can occur due to being forced, or by global state access

  if (cswitch_forced)
    return true;

  // Mutex / condition-var / rwlock / barrier / spinlock accesses should
  // contribute to MPOR dependency tracking (so lock/unlock pairs create
  // dependency chains between threads), but must not by themselves force
  // a context switch point here — they already drive scheduling through
  // the pthread library's explicit switch mechanisms, and treating every
  // lock access as a cswitch point blows up the DFS width.
  auto any_non_sync = [&](const std::set<expr2tc> &s) {
    for (const auto &e : s)
      if (!is_pthread_sync_type(e->type))
        return true;
    return false;
  };

  if (
    any_non_sync(thread_last_reads[active_thread]) ||
    any_non_sync(thread_last_writes[active_thread]))
    return true;
  return false;
}

bool execution_statet::can_execution_continue() const
{
  if (threads_state[active_thread].thread_ended)
    return false;

  if (threads_state[active_thread].call_stack.empty())
    return false;

  return true;
}

std::size_t execution_statet::generate_hash() const
{
  auto l2 = std::dynamic_pointer_cast<state_hashing_level2t>(state_level2);
  assert(l2 != nullptr);

  // State fingerprint for interleaving dedup: combine the L2 value hash with
  // each thread's program-counter location. A size_t structural hash (the
  // same family irep2 uses for hash-consing) replaces the former Boost SHA-1;
  // a collision only over-prunes interleavings, the same risk class crc()
  // already carries tool-wide.
  //
  // The return continuation is part of the fingerprint because a pc does not
  // identify it: one function reached from two call sites stands at one pc with
  // one set of L0 values, and returns to different places. Depth alone does not
  // separate those -- two calls from the same caller sit at equal depth -- so
  // each frame's calling location is mixed in. Without it a bug reachable only
  // past the later state is pruned away in silence.
  //
  // The active thread is part of it for the same reason: equal pcs and equal
  // values still leave two states scheduling differently, because MPOR's
  // dependency chain and decide_ileave_direction's scan both key off which
  // thread just ran. Omitting it collided such pairs, and
  // post_hash_collision_cleanup marks every switch from the survivor explored,
  // so the violating schedule was never generated (#6831).
  std::size_t h = l2->generate_l2_state_hash();
  esbmct::hash_combine(h, active_thread);
  for (const auto &it : threads_state)
  {
    esbmct::hash_combine(h, it.source.pc->location_number);
    for (const auto &frame : it.call_stack)
      esbmct::hash_combine(h, frame.calling_location.pc->location_number);
  }

  return h;
}

std::size_t execution_statet::update_hash_for_assignment(const expr2tc &rhs)
{
  // irep2's cached structural hash (crc) — an atomic size_t on the node,
  // populated by hash-consing — replaces the former per-assignment SHA-1
  // walk of the whole RHS tree, which dominated the --state-hashing path.
  return rhs->crc();
}

void execution_statet::print_stack_traces(unsigned int indent) const
{
  std::vector<goto_symex_statet>::const_iterator it;
  std::string spaces = std::string("");
  unsigned int i;

  for (i = 0; i < indent; i++)
    spaces += " ";

  i = 0;
  for (it = threads_state.begin(); it != threads_state.end(); ++it)
  {
    std::ostringstream oss;
    oss << spaces << "Thread " << i++ << ":"
        << "\n";
    it->print_stack_trace(indent + 2, oss);
    oss << "\n";
    log_status("{}", oss.str());
  }
}

void execution_statet::switch_to_monitor()
{
  if (threads_state[monitor_tid].thread_ended)
  {
    // Warn at most once per process: the message is informational and not
    // tied to which execution_statet noticed the ended monitor.
    static bool warned = false;
    if (!warned)
    {
      log_error(
        "Switching to ended monitor; you need to increase its "
        "context or prefix bound");
      warned = true;
    }

    return;
  }

  assert(tid_is_set && "Must set monitor thread before switching to monitor\n");
  assert(
    !mon_from_tid && "Switching to monitor without having switched away\n");

  monitor_from_tid = active_thread;
  mon_from_tid = true;

  if (monitor_tid != get_active_state_number())
  {
    // Bypass switch_to_thread: a monitor switch must not be followed by
    // update_after_switch_point/execute_guard, and must adopt the source
    // thread's guard instead.
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
  if (threads_state[monitor_tid].thread_ended)
    return;

  assert(tid_is_set && "Must set monitor thread before switching from mon\n");
  assert(mon_from_tid && "Switching from monitor without switching to\n");

  assert(
    monitor_tid == active_thread &&
    "Must call switch_from_monitor from monitor thread\n");

  // Bypass switch_to_thread: a monitor switch must not be followed by
  // update_after_switch_point/execute_guard, and must adopt the monitor
  // thread's guard instead.
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

std::shared_ptr<renaming::level2t>
execution_statet::ex_state_level2t::clone() const
{
  return std::shared_ptr<ex_state_level2t>(new ex_state_level2t(*this));
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
  if (smt_during_symex)
    target->pop_ctx();
}

std::shared_ptr<execution_statet> dfs_execution_statet::clone() const
{
  std::shared_ptr<dfs_execution_statet> d =
    std::make_shared<dfs_execution_statet>(*this);

  // Duplicate target equation; or if we're encoding at runtime, push a context.
  if (smt_during_symex)
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

std::shared_ptr<execution_statet> schedule_execution_statet::clone() const
{
  std::shared_ptr<schedule_execution_statet> s =
    std::make_shared<schedule_execution_statet>(*this);

  // Don't duplicate target equation.
  s->target = target;
  return s;
}

void schedule_execution_statet::claim(
  const expr2tc &expr,
  const std::string &msg)
{
  unsigned int tmp_total, tmp_remaining, tmp_simplified;

  tmp_total = total_claims;
  tmp_remaining = remaining_claims;
  tmp_simplified = simplified_claims;
  execution_statet::claim(expr, msg);

  tmp_total = total_claims - tmp_total;
  tmp_remaining = remaining_claims - tmp_remaining;
  tmp_simplified = simplified_claims - tmp_simplified;

  *ptotal_claims += tmp_total;
  *premaining_claims += tmp_remaining;
  *psimplified_claims += tmp_simplified;
}

execution_statet::state_hashing_level2t::state_hashing_level2t(
  execution_statet &ref)
  : ex_state_level2t(ref)
{
}

std::shared_ptr<renaming::level2t>
execution_statet::state_hashing_level2t::clone() const
{
  return std::make_shared<state_hashing_level2t>(*this);
}

void execution_statet::state_hashing_level2t::make_assignment(
  expr2tc &lhs_sym,
  const expr2tc &const_value,
  const expr2tc &assigned_value)
{
  renaming::level2t::make_assignment(lhs_sym, const_value, assigned_value);

  // If there's no body to the assignment, don't hash.
  if (!is_nil_expr(assigned_value))
  {
    // XXX - consider whether to use l1 names instead. Recursion, reentrancy.
    std::size_t hash = owner->update_hash_for_assignment(assigned_value);
    std::string orig_name = to_symbol2t(lhs_sym).thename.as_string();
    current_hashes[orig_name] = hash;
  }
}

std::size_t
execution_statet::state_hashing_level2t::generate_l2_state_hash() const
{
  // Combine each (variable, value-hash) pair into one size_t fingerprint.
  // current_hashes is an ordered map, so iteration order is stable within a
  // run (the order tracks irep_idt intern indices, so the fingerprint is a
  // valid intra-run dedup key but is not reproducible across runs — same as
  // the former SHA-1-over-this-map, so no behavioural change).
  // Mixing the key as well as the value makes the fingerprint sensitive to
  // *which* variable holds a value (not just the multiset of value hashes),
  // which tightens interleaving dedup — fewer false collisions, less
  // over-pruning.
  std::size_t h = 0;
  for (const auto &current_hashe : current_hashes)
  {
    esbmct::hash_combine(h, current_hashe.first);
    esbmct::hash_combine(h, current_hashe.second);
  }
  return h;
}