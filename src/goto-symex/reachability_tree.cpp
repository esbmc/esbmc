#include <goto-programs/goto_functions.h>
#include <goto-symex/goto_symex.h>
#include <goto-symex/reachability_tree.h>
#include <irep2/irep2_expr.h>
#include <util/config.h>
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
  sym.set_type(bool_typet());
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

  readonly_global_opt =
    options.get_bool_option("cswitch-skip-readonly-globals");
  scan_program_writes();
}

/* White-list of ESBMC internal symbols, mirrors get_expr_globals(). */
static bool is_internal_global_name(const std::string &sn)
{
  return sn == "c:@__ESBMC_alloc" || sn == "c:@__ESBMC_alloc_size" ||
         sn == "c:@__ESBMC_is_dynamic" ||
         sn == "c:@__ESBMC_blocked_threads_count" ||
         sn == "c:@__ESBMC_rounding_mode" ||
         sn.find("c:pthread_lib") != std::string::npos;
}

/* Insert `name` into `out` if it names a storage-bearing user global. */
static void add_if_global(
  const irep_idt &name,
  const namespacet &ns,
  std::unordered_set<irep_idt, irep_id_hash> &out)
{
  const symbolt *s = ns.lookup(name);
  if (!s || is_internal_global_name(name.as_string()))
    return;
  if (s->static_lifetime || s->get_type().is_dynamic_set())
    out.insert(name);
}

/* Walk expression e; any symbol2t that refers to a storage-bearing global is
 * inserted into `out`. A dereference seen as a (sub-)target flips
 * indirect_write: we cannot name the writee, so every address-taken global
 * must thereafter be treated as possibly written (see may_be_written). */
static void collect_write_targets(
  const expr2tc &e,
  const namespacet &ns,
  std::unordered_set<irep_idt, irep_id_hash> &out,
  bool &indirect_write)
{
  if (is_nil_expr(e))
    return;

  if (is_dereference2t(e))
  {
    indirect_write = true;
    return;
  }

  if (is_index2t(e))
    return collect_write_targets(
      to_index2t(e).source_value, ns, out, indirect_write);
  if (is_member2t(e))
    return collect_write_targets(
      to_member2t(e).source_value, ns, out, indirect_write);
  if (is_typecast2t(e))
    return collect_write_targets(
      to_typecast2t(e).from, ns, out, indirect_write);

  if (is_symbol2t(e))
    return add_if_global(to_symbol2t(e).thename, ns, out);

  e->foreach_operand([&](const expr2tc &sub) {
    collect_write_targets(sub, ns, out, indirect_write);
  });
}

/* Descend an lvalue to the named global it denotes, stopping at a dereference
 * (the address of `*p` exposes no statically-named global). */
static void collect_object_globals(
  const expr2tc &e,
  const namespacet &ns,
  std::unordered_set<irep_idt, irep_id_hash> &out)
{
  if (is_nil_expr(e) || is_dereference2t(e))
    return;
  if (is_index2t(e))
    return collect_object_globals(to_index2t(e).source_value, ns, out);
  if (is_member2t(e))
    return collect_object_globals(to_member2t(e).source_value, ns, out);
  if (is_typecast2t(e))
    return collect_object_globals(to_typecast2t(e).from, ns, out);
  if (is_symbol2t(e))
    add_if_global(to_symbol2t(e).thename, ns, out);
}

/* Walk e; for every address_of(obj) sub-expression record the named global
 * whose address escapes. ESBMC lowers all array/function decay into an
 * explicit address_of, so this captures every way a global can enter a
 * pointer's value set. */
static void collect_address_taken(
  const expr2tc &e,
  const namespacet &ns,
  std::unordered_set<irep_idt, irep_id_hash> &out)
{
  if (is_nil_expr(e))
    return;

  if (is_address_of2t(e))
    collect_object_globals(to_address_of2t(e).ptr_obj, ns, out);

  e->foreach_operand([&](const expr2tc &sub) {
    collect_address_taken(sub, ns, out);
  });
}

void reachability_treet::scan_program_writes()
{
  if (!readonly_global_opt)
    return;

  Forall_goto_functions (f_it, goto_functions)
  {
    if (f_it->first == "__ESBMC_main")
      continue;

    if (f_it->second.body.hide)
      continue;

    for (const auto &ins : f_it->second.body.instructions)
    {
      if (is_nil_expr(ins.code))
        continue;

      if (ins.is_assign())
      {
        const code_assign2t &a = to_code_assign2t(ins.code);
        collect_write_targets(
          a.target, ns, ever_written_globals, any_indirect_write);
      }
      else if (ins.is_function_call())
      {
        const code_function_call2t &c = to_code_function_call2t(ins.code);
        if (!is_nil_expr(c.ret))
          collect_write_targets(
            c.ret, ns, ever_written_globals, any_indirect_write);
        // Writes via pointer arguments are picked up when the callee body is
        // scanned (every function body participates in this loop).
      }
    }
  }

  // Second pass: record every global whose address is taken anywhere in the
  // program, scanning all functions — including __ESBMC_main, where file-scope
  // initialisers such as `int *gp = &g;` live. Only an address-taken global
  // can be the target of a write through an unresolved pointer, so this lets
  // never-address-taken globals stay optimisable even when any_indirect_write.
  Forall_goto_functions (f_it, goto_functions)
  {
    for (const auto &ins : f_it->second.body.instructions)
    {
      collect_address_taken(ins.code, ns, address_taken_globals);
      collect_address_taken(ins.guard, ns, address_taken_globals);
    }
  }
}

void reachability_treet::setup_for_new_explore()
{
  std::shared_ptr<symex_targett> targ;

  exploration_frames.clear();

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

  exploration_framet frame;
  frame.state = std::shared_ptr<execution_statet>(s);
  frame.scheduler.reset(s->threads_state.size());
  exploration_frames.push_back(frame);
  cur_frame_it = exploration_frames.begin();
  targ->push_ctx(); // Start with a depth of 1.
}

execution_statet &reachability_treet::get_cur_state()
{
  return *cur_frame_it->state;
}

const execution_statet &reachability_treet::get_cur_state() const
{
  return *cur_frame_it->state;
}

bool reachability_treet::has_more_states()
{
  return !exploration_frames.empty();
}

void reachability_treet::scheduler_framet::ensure_thread_count(
  unsigned int count)
{
  if (explored_threads.size() < count)
    explored_threads.resize(count, false);
}

void reachability_treet::scheduler_framet::reset(unsigned int count)
{
  explored_threads.assign(count, false);
}

void reachability_treet::scheduler_framet::mark_all_explored(unsigned int count)
{
  ensure_thread_count(count);
  std::fill(explored_threads.begin(), explored_threads.end(), true);
}

bool reachability_treet::scheduler_framet::is_explored(unsigned int tid) const
{
  return tid < explored_threads.size() && explored_threads[tid];
}

void reachability_treet::scheduler_framet::mark_explored(unsigned int tid)
{
  ensure_thread_count(tid + 1);
  explored_threads[tid] = true;
}

reachability_treet::scheduler_framet &
reachability_treet::get_cur_scheduler_frame()
{
  return cur_frame_it->scheduler;
}

const reachability_treet::scheduler_framet &
reachability_treet::get_cur_scheduler_frame() const
{
  return cur_frame_it->scheduler;
}

int reachability_treet::get_CS_bound() const
{
  return CS_bound;
}

bool reachability_treet::check_for_hash_collision() const
{
  const execution_statet &ex_state = get_cur_state();

  std::size_t hash = ex_state.generate_hash();
  if (hit_hashes.find(hash) != hit_hashes.end())
    return true;

  return false;
}

void reachability_treet::post_hash_collision_cleanup()
{
  get_cur_scheduler_frame().mark_all_explored(
    get_cur_state().threads_state.size());
}

void reachability_treet::update_hash_collision_set()
{
  execution_statet &ex_state = get_cur_state();

  std::size_t hash = ex_state.generate_hash();
  hit_hashes.insert(hash);
}

void reachability_treet::create_next_state()
{
  execution_statet &ex_state = get_cur_state();

  if (next_thread_id != ex_state.threads_state.size())
  {
    auto new_state = ex_state.clone();
    exploration_frames.push_back({new_state, scheduler_framet{}});

    /* Make it active, make it follow on from previous state... */
    if (new_state->get_active_state_number() != next_thread_id)
      new_state->increment_context_switch();

    new_state->switch_to_thread(next_thread_id);
    new_state->update_after_switch_point();
    exploration_frames.back().scheduler.reset(new_state->threads_state.size());
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
    return check_thread_viable(tid, true) && dfs_explore_thread(tid);
  };

  signed int tid = 0, user_tid = 0;

  // Get thread ID from user if interactive mode is enabled
  get_cur_scheduler_frame().ensure_thread_count(ex_state.threads_state.size());
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
  auto it = cur_frame_it;
  ++it;

  if (it != exploration_frames.end())
  {
    cur_frame_it++;
  }
  else
  {
    if (step_next_state())
    {
      cur_frame_it++;
    }
    else
    {
      if (config.options.get_bool_option("print-stack-traces"))
        print_ileave_trace();
      has_complete_formula = true;
    }
  }
}

void reachability_treet::drain_to_unexplored(bool add_leak_checks)
{
  while (exploration_frames.size() > 0 && !step_next_state())
  {
    // Only fire the leak walker on the final frame when every thread has
    // reached a terminal state — otherwise a still-running spawned thread
    // may hold the only live reference to a dynamic object via its stack
    // frame, which the globals-rooted reachability constraint misses and
    // spuriously reports as forgotten. See #4634.
    if (
      add_leak_checks && exploration_frames.size() == 1 &&
      cur_frame_it->state->all_threads_terminal())
      cur_frame_it->state->add_memory_leak_checks();

    erase_current_frame();
  }

  if (exploration_frames.size() > 0)
    cur_frame_it++;
}

bool reachability_treet::reset_to_unexplored_state()
{
  // After executing up to a point where all threads have ended and returning
  // that equation to the caller, free and remove fully explored execution
  // states back to the point where there's an unexplored one.

  // Eliminate final execution state, then attempt to generate another one from
  // the last on the list. If we can, it's an unexplored state, if we can't,
  // all depths from the current execution state are explored, so delete it.

  erase_current_frame();
  drain_to_unexplored(/*add_leak_checks=*/false);

  if (exploration_frames.size() && !smt_during_symex)
  {
    // When backtracking, erase all the assertions from the equation before
    // continuing forwards. They've all already been checked, in the trace we
    // just backtracked from. Thus there's no point in checking them again.
    symex_target_equationt *eq =
      static_cast<symex_target_equationt *>(cur_frame_it->state->target.get());
    unsigned int num_asserts = eq->clear_assertions();

    // Remove them from the count of remaining assertions to check. This allows
    // for more traces to be discarded because they do not contain any
    // unchecked assertions.
    cur_frame_it->state->total_claims -= num_asserts;
    cur_frame_it->state->remaining_claims -= num_asserts;
  }

  return exploration_frames.size();
}

void reachability_treet::go_next_state()
{
  auto it = cur_frame_it;
  it++;
  if (it != exploration_frames.end())
    cur_frame_it++;
  else
    drain_to_unexplored(/*add_leak_checks=*/true);
}

void reachability_treet::print_ileave_trace() const
{
  int i = 0;

  log_status("Context switch trace for interleaving:");
  for (const auto &frame : exploration_frames)
  {
    log_status("Context switch point {}", i);
    frame.state->print_stack_traces(4);
    ++i;
  }
}

bool reachability_treet::check_thread_viable(unsigned int tid, bool quiet) const
{
  const execution_statet &ex = get_cur_state();

  if (get_cur_scheduler_frame().is_explored(tid))
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

void reachability_treet::mark_active_thread_explored()
{
  get_cur_scheduler_frame().mark_explored(get_cur_state().active_thread);
}

bool reachability_treet::dfs_explore_thread(unsigned int tid)
{
  scheduler_framet &frame = get_cur_scheduler_frame();
  if (frame.is_explored(tid))
    return false;

  if (get_cur_state().threads_state.at(tid).call_stack.empty())
    return false;

  if (get_cur_state().threads_state.at(tid).thread_ended)
    return false;

  frame.mark_explored(tid);
  return true;
}

void reachability_treet::erase_current_frame()
{
  if (exploration_frames.empty())
    return;

  if (cur_frame_it == exploration_frames.begin())
  {
    cur_frame_it = exploration_frames.erase(cur_frame_it);
    return;
  }

  auto prev = std::prev(cur_frame_it);
  exploration_frames.erase(cur_frame_it);
  cur_frame_it = prev;
}

goto_symext::symex_resultt reachability_treet::get_next_formula()
{
  assert(!exploration_frames.empty() && "Must setup RT before exploring");

  while (!is_has_complete_formula())
  {
    while ((!get_cur_state().has_cswitch_point_occured() ||
            get_cur_state().check_if_ileaves_blocked()) &&
           get_cur_state().can_execution_continue())
      get_cur_state().symex_step(*this);

    if (por)
    {
      get_cur_state().calculate_mpor_constraints();
      if (get_cur_state().is_transition_blocked_by_mpor())
        break;
    }

    if (state_hashing)
    {
      if (check_for_hash_collision())
      {
        post_hash_collision_cleanup();
        break;
      }

      update_hash_collision_set();
    }
    next_thread_id = decide_ileave_direction(get_cur_state());

    if (
      get_cur_state().interleaving_unviable &&
      next_thread_id != get_cur_state().active_thread)
      break;
    create_next_state();

    switch_to_next_execution_state();
  }

  // Only fire the leak walker on schedules whose tail is a genuine program
  // termination — every thread terminal. Otherwise a still-running spawned
  // thread may hold the only live reference to a dynamic object via its
  // stack frame; the leak walker's reachability constraint is rooted at
  // globals only, so it would spuriously report that object forgotten. The
  // DFS will produce the all-threads-terminal sibling schedule separately,
  // and the check fires correctly there. See #4634.
  if (cur_frame_it->state->all_threads_terminal())
    cur_frame_it->state->add_memory_leak_checks();

  has_complete_formula = false;

  return get_cur_state().get_symex_result();
}

bool reachability_treet::setup_next_formula()
{
  return reset_to_unexplored_state();
}

goto_symext::symex_resultt reachability_treet::generate_schedule_formula()
{
  while (has_more_states())
  {
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
