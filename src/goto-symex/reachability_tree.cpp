#include <goto-symex/goto_symex.h>
#include <goto-symex/reachability_tree.h>
#include <util/config.h>
#include <util/crypto_hash.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/message.h>
#include <util/std_expr.h>

reachability_treet::reachability_treet(
  goto_functionst &goto_functions,
  const namespacet &ns,
  optionst &opts,
  std::shared_ptr<symex_targett> target,
  contextt &context)
  : goto_functions(goto_functions),
    permanent_context(context),
    ns(ns),
    options(opts)
{
  // Put a few useful symbols in the symbol table.
  symbolt sym;
  sym.type = bool_typet();
  sym.id = "execution_statet::\\guard_exec";
  sym.name = "execution_statet::\\guard_exec";
  context.move(sym);

  CS_bound = atoi(options.get_option("context-bound").c_str());
  TS_slice = atoi(options.get_option("time-slice").c_str());
  state_hashing = options.get_bool_option("state-hashing");
  directed_interleavings = options.get_bool_option("direct-interleavings");
  interactive_ileaves = options.get_bool_option("interactive-ileaves");
  schedule = options.get_bool_option("schedule");
  smt_during_symex = options.get_bool_option("smt-during-symex");
  por = !options.get_bool_option("no-por");
  main_thread_ended = false;
  target_template = std::move(target);
}

void reachability_treet::setup_for_new_explore()
{
  std::shared_ptr<symex_targett> targ;

  execution_states.clear();

  has_complete_formula = false;

  execution_statet *s;
  if (schedule)
  {
    schedule_target = target_template->clone();
    targ = schedule_target;
    s = reinterpret_cast<execution_statet *>(new schedule_execution_statet(
      goto_functions,
      ns,
      this,
      schedule_target,
      permanent_context,
      options,
      &schedule_total_claims,
      &schedule_remaining_claims,
      &schedule_simplified_claims));
  }
  else
  {
    targ = target_template->clone();
    s = reinterpret_cast<execution_statet *>(new dfs_execution_statet(
      goto_functions, ns, this, targ, permanent_context, options));
    schedule_target = nullptr;
  }

  execution_states.emplace_back(s);
  cur_state_it = execution_states.begin();
  targ->push_ctx(); // Start with a depth of 1.
}

execution_statet &reachability_treet::get_cur_state()
{
  return **cur_state_it;
}

const execution_statet &reachability_treet::get_cur_state() const
{
  return **cur_state_it;
}

bool reachability_treet::has_more_states()
{
  return execution_states.size() > 0;
}

int reachability_treet::get_CS_bound() const
{
  return CS_bound;
}

bool reachability_treet::check_for_hash_collision() const
{
  const execution_statet &ex_state = get_cur_state();

  crypto_hash hash;
  hash = ex_state.generate_hash();
  if (hit_hashes.find(hash) != hit_hashes.end())
    return true;

  return false;
}

void reachability_treet::post_hash_collision_cleanup()
{
  for (auto &&it : get_cur_state().DFS_traversed)
    it = true;
}

void reachability_treet::update_hash_collision_set()
{
  execution_statet &ex_state = get_cur_state();

  crypto_hash hash;
  hash = ex_state.generate_hash();
  hit_hashes.insert(hash);
}

void reachability_treet::create_next_state()
{
  execution_statet &ex_state = get_cur_state();

  if (next_thread_id != ex_state.threads_state.size())
  {
    auto new_state = ex_state.clone();
    execution_states.push_back(new_state);

    /* Make it active, make it follow on from previous state... */
    if (new_state->get_active_state_number() != next_thread_id)
      new_state->increment_context_switch();

    new_state->switch_to_thread(next_thread_id);
    new_state->update_after_switch_point();
  }
}

bool reachability_treet::step_next_state()
{
  next_thread_id = decide_ileave_direction(get_cur_state());
  if (next_thread_id != get_cur_state().threads_state.size())
  {
    create_next_state();
    return true;
  }

  return false;
}

unsigned int
reachability_treet::decide_ileave_direction(execution_statet &ex_state)
{
  auto is_thread_schedulable = [&](int tid) {
    return check_thread_viable(tid, true) && ex_state.dfs_explore_thread(tid);
  };

  signed int tid = 0, user_tid = 0;

  // Get thread ID from user if interactive mode is enabled
  tid = get_cur_state().active_thread + 1;
  if (interactive_ileaves)
  {
    tid = get_ileave_direction_from_user();
    user_tid = tid;
  }

  // Try finding a schedulable thread in the forward direction
  for (; tid < (int)ex_state.threads_state.size(); ++tid)
  {
    if (is_thread_schedulable(tid))
      break;
  }

  // If no thread was found, search in the reverse direction
  if (tid == (int)ex_state.threads_state.size())
  {
    for (tid = get_cur_state().active_thread; tid >= 0; --tid)
    {
      if (is_thread_schedulable(tid))
        break;
    }
  }

  // If no valid thread is found, set tid to the size of threads_state
  if (tid < 0)
    tid = ex_state.threads_state.size();

  // Validate user choice in interactive mode
  if (interactive_ileaves && tid != user_tid)
  {
    log_error("Ileave code selected different thread from user choice");
    abort();
  }

  return tid;
}

bool reachability_treet::is_has_complete_formula()
{
  return has_complete_formula;
}

void reachability_treet::switch_to_next_execution_state()
{
  std::list<std::shared_ptr<execution_statet>>::iterator it = cur_state_it;
  ++it;

  if (it != execution_states.end())
  {
    cur_state_it++;
  }
  else
  {
    if (step_next_state())
    {
      cur_state_it++;
    }
    else
    {
      if (config.options.get_bool_option("print-stack-traces"))
        print_ileave_trace();
      has_complete_formula = true;
    }
  }
}

bool reachability_treet::reset_to_unexplored_state()
{
  // After executing up to a point where all threads have ended and returning
  // that equation to the caller, free and remove fully explored execution
  // states back to the point where there's an unexplored one.

  // Eliminate final execution state, then attempt to generate another one from
  // the last on the list. If we can, it's an unexplored state, if we can't,
  // all depths from the current execution state are explored, so delete it.

  auto it = cur_state_it--;
  execution_states.erase(it);

  while (execution_states.size() > 0 && !step_next_state())
  {
    it = cur_state_it--;
    execution_states.erase(it);
  }

  if (execution_states.size() > 0)
    cur_state_it++;

  if (execution_states.size() && !smt_during_symex)
  {
    // When backtracking, erase all the assertions from the equation before
    // continuing forwards. They've all already been checked, in the trace we
    // just backtracked from. Thus there's no point in checking them again.
    symex_target_equationt *eq =
      static_cast<symex_target_equationt *>((*cur_state_it)->target.get());
    unsigned int num_asserts = eq->clear_assertions();

    // Remove them from the count of remaining assertions to check. This allows
    // for more traces to be discarded because they do not contain any
    // unchecked assertions.
    (*cur_state_it)->total_claims -= num_asserts;
    (*cur_state_it)->remaining_claims -= num_asserts;
  }

  return execution_states.size();
}

void reachability_treet::go_next_state()
{
  std::list<std::shared_ptr<execution_statet>>::iterator it = cur_state_it;
  it++;
  if (it != execution_states.end())
    cur_state_it++;
  else
  {
    while (execution_states.size() > 0 && !step_next_state())
    {
      it = cur_state_it;
      cur_state_it--;

      // For the last one:
      if (execution_states.size() == 1)
        (*it)->add_memory_leak_checks();

      execution_states.erase(it);
    }

    if (execution_states.size() > 0)
      cur_state_it++;
  }
}

void reachability_treet::print_ileave_trace() const
{
  std::list<std::shared_ptr<execution_statet>>::const_iterator it;
  int i = 0;

  log_status("Context switch trace for interleaving:");
  for (it = execution_states.begin(); it != execution_states.end(); ++it, ++i)
  {
    log_status("Context switch point {}", i);
    (*it)->print_stack_traces(4);
  }
}

bool reachability_treet::check_thread_viable(unsigned int tid, bool quiet) const
{
  const execution_statet &ex = get_cur_state();

  if (ex.DFS_traversed.at(tid) == true)
  {
    if (!quiet)
      log_status("Thread unschedulable as it's already been explored");
    return false;
  }

  if (ex.threads_state.at(tid).call_stack.empty())
  {
    if (!quiet)
      log_status("Thread unschedulable due to empty call stack");
    return false;
  }

  if (ex.threads_state.at(tid).thread_ended)
  {
    if (!quiet)
      log_status("That thread has ended");
    return false;
  }

#if 0
  if (por && !ex.is_thread_mpor_schedulable(tid)) {
    if (!quiet)
      log_status("Thread unschedulable due to POR");
    return false;
  }
#endif

  if (ex.tid_is_set && ex.monitor_tid == tid)
  {
    if (!quiet)
      log_status("Can't context switch to a monitor thread");
    return false;
  }

  return true;
}

goto_symext::symex_resultt reachability_treet::get_next_formula()
{
  assert(execution_states.size() > 0 && "Must setup RT before exploring");

  while (!is_has_complete_formula())
  {
    while ((!get_cur_state().has_cswitch_point_occured() ||
            get_cur_state().check_if_ileaves_blocked()) &&
           get_cur_state().can_execution_continue())
      get_cur_state().symex_step(*this);

    if (state_hashing)
    {
      if (check_for_hash_collision())
      {
        post_hash_collision_cleanup();
        break;
      }

      update_hash_collision_set();
    }

    if (por)
    {
      get_cur_state().calculate_mpor_constraints();
      if (get_cur_state().is_transition_blocked_by_mpor())
        break;
    }

    next_thread_id = decide_ileave_direction(get_cur_state());

    if (
      get_cur_state().interleaving_unviable &&
      next_thread_id != get_cur_state().active_thread)
      break;
    create_next_state();

    switch_to_next_execution_state();
  }

  (*cur_state_it)->add_memory_leak_checks();

  has_complete_formula = false;

  return get_cur_state().get_symex_result();
}

bool reachability_treet::setup_next_formula()
{
  return reset_to_unexplored_state();
}

goto_symext::symex_resultt reachability_treet::generate_schedule_formula()
{
  int total_states = 0;
  while (has_more_states())
  {
    total_states++;
    while ((!get_cur_state().has_cswitch_point_occured() ||
            get_cur_state().check_if_ileaves_blocked()) &&
           get_cur_state().can_execution_continue())
    {
      get_cur_state().symex_step(*this);
    }

    if (state_hashing)
    {
      if (check_for_hash_collision())
      {
        post_hash_collision_cleanup();
        go_next_state();
        continue;
      }

      update_hash_collision_set();
    }

    next_thread_id = decide_ileave_direction(get_cur_state());

    create_next_state();

    go_next_state();
  }

  return goto_symext::symex_resultt(
    schedule_target,
    schedule_total_claims,
    schedule_remaining_claims,
    schedule_simplified_claims);
}
