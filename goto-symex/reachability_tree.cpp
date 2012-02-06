/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

/* Byte order includes, for context switch checkpoint files */
#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#else
#include <winsock2.h>
#undef small // The mingw32 headers are /absolutely rubbish/, or perhaps the
             // windows headers by themselves.
#endif

#include "reachability_tree.h"
#include "goto_symex.h"
#include <i2string.h>
#include <expr_util.h>
#include <std_expr.h>
#include <config.h>

#include "crypto_hash.h"

/*******************************************************************
 Function: reachability_treet::get_cur_state

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

execution_statet & reachability_treet::get_cur_state()
{

  return **cur_state_it;
}

const execution_statet & reachability_treet::get_cur_state() const
{

  return **cur_state_it;
}

/*******************************************************************
 Function: reachability_treet::has_more_states

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::has_more_states()
{
  return execution_states.size() > 0;
}

/*******************************************************************
 Function: reachability_treet::check_CS_bound

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::check_CS_bound()
{
  if(CS_bound  != -1 && get_cur_state().get_context_switch() >= CS_bound)
  {
    return true;
  }
  else
    return false;
}

/*******************************************************************
 Function: reachability_treet::get_CS_bound

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

int reachability_treet::get_CS_bound()
{
  return CS_bound;
}

/*******************************************************************
 Function: reachability_treet::is_global_assign

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::is_global_assign(const exprt &code)
{

  int num_read_globals = get_cur_state().get_expr_read_globals(ns,code.op1());

  if (num_read_globals)
	return true;
  else
	return false;
}

/*******************************************************************
 Function: reachability_treet::generate_states_before_read

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::generate_states_before_read(const exprt &code)
{

  if (check_CS_bound())
    return false;

  if (get_cur_state().get_active_atomic_number() > 0)
   	return false;

  if (get_cur_state().get_expr_read_globals(ns,code) > 0)
    return generate_states_base(code);
  else
    return false;
}

/*******************************************************************

 Function: reachability_treet::generate_states_before_assign

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::generate_states_before_assign(const exprt &code, execution_statet &ex_state)
{

  if(code.operands().size()!=2)
    throw "assignment expects two operands";

  if(check_CS_bound())
    return false;

  if(get_cur_state().get_active_atomic_number() > 0)
    return false;

  int num_write_globals = get_cur_state().get_expr_write_globals(ns,code.op0());
  int num_read_globals = get_cur_state().get_expr_read_globals(ns,code.op1());

  if(num_read_globals + num_write_globals > 0)
  {
	ex_state.reexecute_instruction = false;
    return generate_states_base(code);
  }

  return false;
}

/*******************************************************************

 Function: reachability_treet::generate_states

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::generate_states()
{

  if(check_CS_bound())
    return false;

  if(get_cur_state().get_active_atomic_number() > 0)
  	return false;

  // do analysis here
  return generate_states_base(exprt());
}

/*******************************************************************
 Function: reachability_treet::apply_static_por

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::apply_static_por(const execution_statet &ex_state, const exprt &expr, int i) const
{
  bool consider = true;

  if (por)
  {
    if(ex_state.last_global_expr.is_not_nil() && !expr.id().empty())
    {
      if(i < ex_state._active_thread)
      {
        if(ex_state.last_global_read_write.write_set.empty() &&
           ex_state._exprs_read_write.at(i+1).write_set.empty() &&
           ex_state._exprs_read_write.at(ex_state._active_thread).write_set.empty())
        {
          return false;
        }

        consider = false;

        if(ex_state.last_global_read_write.has_write_intersect(ex_state._exprs_read_write.at(i+1).write_set))
        {
          consider = true;
        }
        else if(ex_state.last_global_read_write.has_write_intersect(ex_state._exprs_read_write.at(i+1).read_set))
        {
          consider = true;
        }
        else if(ex_state.last_global_read_write.has_read_intersect(ex_state._exprs_read_write.at(i+1).write_set))
        {
          consider = true;
        }
      }
    }
  }

  return consider;
}

/*******************************************************************
 Function: reachability_treet::generate_states_base

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::generate_states_base(const exprt &expr)
{

  if(CS_bound  != -1 && get_cur_state().get_context_switch() >= CS_bound)
    return false;

  if (directed_interleavings)
    // Don't generate interleavings automatically - instead, the user will
    // inserts intrinsics identifying where they want interleavings to occur,
    // and to what thread.
    return false;

  execution_statet &ex_state = get_cur_state();

  ex_state._exprs.at(ex_state._active_thread) = expr;

  // force the new threads continue execute to visible instruction
  if(ex_state.generating_new_threads > 0)
  {
    /* jmorse - just sets some internal fields, last active, etc */
    ex_state.set_active_state(ex_state.generating_new_threads);
    ex_state.generating_new_threads = -1;
    ex_state.reexecute_instruction = false;
    return true;
  }
  if(ex_state.reexecute_instruction)
  {
    ex_state.reexecute_instruction = false;
    return false;
  }
  if(ex_state.generating_new_threads == -1)
  {
    ex_state.generating_new_threads = 0;
    ex_state.set_active_state(ex_state._last_active_thread);
  }

  if(ex_state.threads_state.size() < 2)
  {
    return false;
  }

  crypto_hash hash;
  if (state_hashing) {
    goto_programt::const_targett pc = ex_state.get_active_state().source.pc;
    hash = ex_state.generate_hash();
    if (hit_hashes.find(hash) != hit_hashes.end())
      return false;
  }

  //new
#if 1
  if(expr.is_not_nil())
  {
    ex_state.last_global_expr = ex_state._exprs.at(ex_state._active_thread);
    ex_state.last_global_read_write = ex_state._exprs_read_write.at(ex_state._active_thread);
  }
#endif

  unsigned int tid = 0, user_tid = 0;

  if (config.options.get_bool_option("interactive-ileaves")) {
    user_tid = tid = get_ileave_direction_from_user(expr);
  }
  //begin - H.Savino
  else if (config.options.get_bool_option("round-robin")) {

    user_tid = tid = get_ileave_direction_from_scheduling(expr);
    if(tid != ex_state._active_thread){
        ex_state._DFS_traversed.at(ex_state._active_thread)=true;
    }
    if(tid == ex_state._active_thread){
        for(tid=0; tid < ex_state.threads_state.size(); tid++)
        {
          if(tid==user_tid)
              continue;
          if(ex_state._DFS_traversed.at(tid))
            continue;
          ex_state._DFS_traversed.at(tid) = true;
        }
       tid=user_tid;
    }
  }
  //end - H.Savino

  for(; tid < ex_state.threads_state.size(); tid++)
  {
    /* For all threads: */

    if(ex_state._DFS_traversed.at(tid))
      continue;

    ex_state._DFS_traversed.at(tid) = true;

    /* Presumably checks whether this thread isn't in user code yet? */
    if(ex_state.threads_state.at(tid).call_stack.empty())
      continue;

    /* Is it even still running? */
    if(ex_state.threads_state.at(tid).thread_ended)
      continue;

    //apply static partial-order reduction
    if(!apply_static_por(ex_state, expr, tid))
      continue;

    break;
  }

  if (config.options.get_bool_option("interactive-ileaves") && tid != user_tid){
    std::cerr << "Ileave code selected different thread from user choice";
    std::cerr << std::endl;
  }

  at_end_of_run = true;

  if (tid != ex_state.threads_state.size()) {

    /* Generate a new execution state, duplicate of previous? */
    execution_statet *new_state = new execution_statet(ex_state);
    execution_states.push_back(new_state);

    //begin - H.Savino
    if (config.options.get_bool_option("round-robin")){
        if(tid == ex_state._active_thread)
            new_state->increment_time_slice();
        else
            new_state->reset_time_slice();
    }
    //end - H.Savino

    /* Make it active, make it follow on from previous state... */
    if (new_state->get_active_state_number() != tid) {
      new_state->increment_context_switch();
      new_state->set_active_state(tid);
    }

    new_state->set_parent_guard(ex_state.get_guard_identifier());
    new_state->reexecute_instruction = true;

//    execution_states.rbegin()->copy_level2_from(ex_state);
//    Copy constructor should duplicate level2 object
    /* Reset interleavings (?) investigated in this new state */
    new_state->reset_DFS_traversed();

    return true;
  } else {
    /* Once we've generated all interleavings from this state, increment hit
     * count so that we don't come back here again */
    if (state_hashing)
      hit_hashes.insert(hash);

    return false;
  }
}

/*******************************************************************
 Function: reachability_treet::is_at_end_of_run

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::is_at_end_of_run()
{

  return at_end_of_run ||
         get_cur_state().get_active_state().thread_ended ||
         get_cur_state().get_active_state().call_stack.empty();
}

/*******************************************************************
 Function: reachability_treet::is_has_complete_formula

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::is_has_complete_formula()
{

  return has_complete_formula;
}

/*******************************************************************
 Function: reachability_treet::switch_to_next_execution_state

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void reachability_treet::switch_to_next_execution_state()
{

  std::list<execution_statet*>::iterator it = cur_state_it;
  it++;

  if(it != execution_states.end()) {
    cur_state_it++;
  } else {
    if (generate_states_base(exprt()))
      cur_state_it++;
    else
      has_complete_formula = true;
  }

  at_end_of_run = false;
}

bool reachability_treet::reset_to_unexplored_state()
{
  std::list<execution_statet*>::iterator it;

  // After executing up to a point where all threads have ended and returning
  // that equation to the caller, free and remove fully explored execution
  // states back to the point where there's an unexplored one.

  // Eliminate final execution state, then attempt to generate another one from
  // the last on the list. If we can, it's an unexplored state, if we can't,
  // all depths from the current execution state are explored, so delete it.

  it = cur_state_it--;
  delete *it;
  execution_states.erase(it);

  while(execution_states.size() > 0 && !generate_states_base(exprt())) {
    it = cur_state_it--;
    delete *it;
    execution_states.erase(it);
  }

  if (execution_states.size() > 0)
    cur_state_it++;

  at_end_of_run = false;
  return execution_states.size() != 0;
}

/*******************************************************************
 Function: reachability_treet::go_next_state

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void reachability_treet::go_next_state()
{

  std::list<execution_statet*>::iterator it = cur_state_it;
  it++;
  if(it != execution_states.end())
    cur_state_it++;
  else
  {
    while(execution_states.size() > 0 && !generate_states_base(exprt()))
    {
      it = cur_state_it;
      cur_state_it--;
      execution_states.erase(it);
    }

    if(execution_states.size() > 0)
      cur_state_it++;
  }

  at_end_of_run = false;
}

reachability_treet::dfs_position::dfs_position(const reachability_treet &rt)
{
  std::list<execution_statet*>::const_iterator it;

  // Iterate through each position in the DFS tree recording data into this
  // object.
  for (it = rt.execution_states.begin(); it != rt.execution_states.end();it++){
    reachability_treet::dfs_position::dfs_state state;
    execution_statet *ex = *it;
    state.location_number = ex->get_active_state().source.pc->location_number;
    state.num_threads = ex->threads_state.size();
    state.explored = ex->_DFS_traversed;

    // The thread taken in this DFS path isn't decided at this execution state,
    // instead it's whatever thread is active in the /next/ state. So, take the
    // currently active thread no and assign it to the previous dfs state
    // we recorded.
    if (states.size() > 0)
      states.back().cur_thread = ex->get_active_state_number();

    states.push_back(state);
  }

  // The final execution state in a DFS is a dummy, there are no paths from it,
  // so assign a dummy cur_thread value.
  states.back().cur_thread = 0;

  checksum = 0; // Use this in the future.
  ileaves = 0; // Can use this depending on a future refactor.
}

reachability_treet::dfs_position::dfs_position(const std::string filename)
{

  read_from_file(filename);
}

const uint32_t reachability_treet::dfs_position::file_magic = 'ECHK';

bool reachability_treet::dfs_position::write_to_file(
                                       const std::string filename) const
{
  uint8_t buffer[8192];
  reachability_treet::dfs_position::file_hdr hdr;
  reachability_treet::dfs_position::file_entry entry;
  std::vector<bool>::const_iterator ex_it;
  std::vector<reachability_treet::dfs_position::dfs_state>::const_iterator it;
  FILE *f;
  unsigned int i;

  f = fopen(filename.c_str(), "wb");
  if (f == NULL) {
    std::cerr << "Couldn't open checkpoint output file" << std::endl;
    return true;
  }

  hdr.magic = htonl(file_magic);
  hdr.checksum = 0;
  hdr.num_states = htonl(states.size());
  hdr.num_ileaves = 0;

  if (fwrite(&hdr, sizeof(hdr), 1, f) != 1)
    goto fail;

  for (it = states.begin(); it != states.end(); it++) {
    entry.location_number = htonl(it->location_number);
    entry.num_threads = htons(it->num_threads);
    entry.cur_thread = htons(it->cur_thread);
    
    if (fwrite(&entry, sizeof(entry), 1, f) != 1)
      goto fail;

    assert(it->explored.size() < 65536);
    assert(it->explored.size() == ntohs(entry.num_threads));

    i = 0;
    memset(buffer, 0, sizeof(buffer));
    for (ex_it = it->explored.begin(); ex_it != it->explored.end(); ex_it++) {
      if (*ex_it) {
        buffer[i >> 3] |= (1 << i & 7);
      }
      i++;
    }

    // Round up
    i += 7;
    i >>= 3;

    assert(i != 0); // Always at least one thread in _existance_.
    if (fwrite(buffer, i, 1, f) != 1)
      goto fail;
  }

  fclose(f);
  return false;

fail:
  std::cerr << "Write error writing checkpoint file" << std::endl;
  fclose(f);
  return true;
}

bool reachability_treet::dfs_position::read_from_file(
                                       const std::string filename)
{
  reachability_treet::dfs_position::file_hdr hdr;
  reachability_treet::dfs_position::file_entry entry;
  FILE *f;
  unsigned int i, j;
  char c;

  f = fopen(filename.c_str(), "rb");
  if (f == NULL) {
    std::cerr << "Couldn't open checkpoint input file" << std::endl;
    return true;
  }

  if (fread(&hdr, sizeof(hdr), 1, f) != 1)
    goto fail;

  if (hdr.magic != htonl(file_magic)) {
    std::cerr << "Magic number indicates that this isn't a checkpoint file"
              << std::endl;
    fclose(f);
    return true;
  }

  for (i = 0; i < ntohl(hdr.num_states); i++) {
    reachability_treet::dfs_position::dfs_state state;
    if (fread(&entry, sizeof(entry), 1, f) != 1)
      goto fail;

    state.location_number = ntohl(entry.location_number);
    state.num_threads = ntohs(entry.num_threads);
    state.cur_thread = ntohs(entry.cur_thread);

    assert(state.num_threads < 65536);
    if (state.cur_thread >= state.num_threads) {
      std::cerr << "Inconsistent checkpoint data" << std::endl;
      fclose(f);
      return true;
    }

    for (j = 0; j < state.num_threads; j++) {
      if (j % 8 == 0) {
        if (fread(&c, sizeof(c), 1, f) != 1)
          goto fail;
      }

      state.explored.push_back(c & (1 << (j & 7)));
    }

    states.push_back(state);
  }

  fclose(f);
  return false;

fail:
  std::cerr << "Read error on checkpoint file" << std::endl;
  fclose(f);
  return true;
}

void
reachability_treet::print_ileave_trace(void) const
{
  std::list<execution_statet*>::const_iterator it;
  int i = 0;

  std::cout << "Context switch trace for interleaving:" << std::endl;
  for (it = execution_states.begin(); it != execution_states.end(); it++, i++) {
    std::cout << "Context switch point " << i << std::endl;
    (*it)->print_stack_traces(ns, 4);
  }
}

int
reachability_treet::get_ileave_direction_from_user(const exprt &expr) const
{
  std::string input;
  unsigned int tid;

  // If the guard on this execution trace is false, no context switches are
  // going to be run over in the future and just general randomness is going to
  // occur. So there's absolutely no reason exploring further.
  if (get_cur_state().get_active_state().guard.is_false()) {
    std::cout << "This trace's guard is false; it will not be evaulated." << std::endl;
    exit(1);
  }

  // First of all, are there actually any valid context switch targets?
  for (tid = 0; tid < get_cur_state().threads_state.size(); tid++) {
    if (check_thread_viable(tid, expr, true))
      break;
  }

  // If no threads were viable, don't present a choice.
  if (tid == get_cur_state().threads_state.size())
    return get_cur_state().threads_state.size();

  std::cout << "Context switch point encountered; please select a thread to run" << std::endl;
  std::cout << "Current thread states:" << std::endl;
  execution_states.back()->print_stack_traces(ns, 4);

  while (std::cout << "Input: ", std::getline(std::cin, input)) {
    if (input == "b") {
      std::cout << "Back unimplemented" << std::endl;
    } else if (input == "q") {
      exit(1);
    } else if (input.size() <= 0) {
      ;
    } else {
      const char *start;
      char *end;
      start = input.c_str();
      tid = strtol(start, &end, 10);
      if (start == end) {
        std::cout << "Not a valid input" << std::endl;
      } else if (tid >= get_cur_state().threads_state.size()) {
        std::cout << "Number out of range";
      } else {
        if (check_thread_viable(tid, expr, false))
          break;
      }
    }
  }

  if (std::cin.eof()) {
    std::cout << std::endl;
    exit(1);
  }

  return tid;
}

//begin - H.Savino
int
reachability_treet::get_ileave_direction_from_scheduling(const exprt &expr) const
{
  unsigned int tid;

    // If the guard on this execution trace is false, no context switches are
    // going to be run over in the future and just general randomness is going to
    // occur. So there's absolutely no reason exploring further.
    if ((expr.operands().size() > 0) &&
      get_cur_state().get_active_state().guard.is_false()) {
          std::cout << "This trace's guard is false; it will not be evaulated." << std::endl;
          exit(1);
    }

    // First of all, are there actually any valid context switch targets?
    for (tid = 0; tid < get_cur_state().threads_state.size(); tid++) {
      if (check_thread_viable(tid, expr, true))
        break;
    }

    // If no threads were viable, don't present a choice.
    if (tid == get_cur_state().threads_state.size())
      return get_cur_state().threads_state.size();

  tid=get_cur_state()._active_thread;

  if(get_cur_state()._TS_number < this->_TS_slice-1){
      if (check_thread_viable(tid, expr, true))
          return tid;
  }
      while(1){
        tid=(tid + 1)%get_cur_state().threads_state.size();
        if (check_thread_viable(tid, expr, true)){
            break;
        }
      }
  return tid;
}
//end - H.Savino

bool
reachability_treet::check_thread_viable(int tid, const exprt &expr, bool quiet) const
{
  const execution_statet &ex = get_cur_state();

  if (ex._DFS_traversed.at(tid) == true) {
    if (!quiet)
      std::cout << "Thread unschedulable as it's already been explored" << std::endl;
    return false;
  }

  if (ex.threads_state.at(tid).call_stack.empty()) {
    if (!quiet)
      std::cout << "Thread unschedulable due to empty call stack" << std::endl;
    return false;
  }

  if (ex.threads_state.at(tid).thread_ended) {
    if (!quiet)
      std::cout << "That thread has ended" << std::endl;
    return false;
  }

  if (!apply_static_por(ex, expr, tid)) {
    if (!quiet)
      std::cout << "Thread unschedulable due to POR" << std::endl;
    return false;
  }

  return true;
}

goto_symext::symex_resultt *
reachability_treet::get_next_formula()
{

  get_cur_state().execute_guard(ns);
  while(!is_has_complete_formula())
  {
    while (!is_at_end_of_run())
      get_cur_state().symex_step(goto_functions, *this);

    switch_to_next_execution_state();
  }

  has_complete_formula = false;

  return get_cur_state().get_symex_result();
}

bool
reachability_treet::setup_next_formula(void)
{

  return reset_to_unexplored_state();
}

goto_symext::symex_resultt *
reachability_treet::generate_schedule_formula()
{

  int total_states = 0;
  while (has_more_states())
  {
    total_states++;
    get_cur_state().execute_guard(ns);
    while (!is_at_end_of_run())
    {
      get_cur_state().symex_step(goto_functions, *this);
    }

    go_next_state();
  }

  return get_cur_state().get_symex_result();
}

bool
reachability_treet::restore_from_dfs_state(void *_dfs)
{
  std::vector<reachability_treet::dfs_position::dfs_state>::const_iterator it;
  unsigned int i;

  const reachability_treet::dfs_position *foo = (const reachability_treet::dfs_position*)_dfs;
  const reachability_treet::dfs_position &dfs = *foo;
  // Symex repeatedly until context switch points. At each point, verify that it
  // happened where we expected it to, and then switch to the correct thread for
  // the history we've been provided with.
  for (it = dfs.states.begin(), i = 0; it != dfs.states.end(); it++, i++) {

    at_end_of_run = false;

    while (!is_at_end_of_run()) {
      // Restore the DFS exploration space so that when an interleaving occurs
      // we take the option leading to the thread we desire to run. This
      // assumes that the DFS exploration path algorithm never changes.
      // Has to occur here; between generating new threads, ESBMC messes with
      // the dfs state.
      for (int dfspos = 0; dfspos < get_cur_state()._DFS_traversed.size();
           dfspos++)
        get_cur_state()._DFS_traversed[dfspos] = true;
      get_cur_state()._DFS_traversed[it->cur_thread] = false;

      get_cur_state().symex_step(goto_functions, *this);
    }
    get_cur_state()._DFS_traversed = it->explored;

    if (get_cur_state().threads_state.size() != it->num_threads) {
      std::cerr << "Unexpected number of threads when reexploring checkpoint"
                << std::endl;
      abort();
    }

    switch_to_next_execution_state();

    // check we're on the right thread; except on the last run, where there are
    // no more threads to be run.
    if (i + 1 < dfs.states.size())
      assert(get_cur_state().get_active_state_number() == it->cur_thread);

#if 0
// XXX jmorse: can't quite get these sequence numbers to line up when they're
// replayed.
    if (get_cur_state().get_active_state().source.pc->location_number !=
        it->location_number) {
      std::cerr << "Interleave at unexpected location when restoring checkpoint"
                << std::endl;
      abort();
    }
#endif
  }

  return false;
}

void reachability_treet::save_checkpoint(const std::string fname) const
{

  reachability_treet::dfs_position pos(*this);
  if (pos.write_to_file(fname))
    std::cerr << "Couldn't save checkpoint; continuing" << std::endl;

  return;
}
