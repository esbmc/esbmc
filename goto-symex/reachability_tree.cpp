/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <arpa/inet.h>

#include <netinet/in.h>

#include "reachability_tree.h"
#include <i2string.h>
#include <expr_util.h>
#include <std_expr.h>
#include <config.h>

#include "crypto_hash.h"

//#define DEBUG

/*******************************************************************
 Function: reachability_treet::get_cur_state

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

execution_statet & reachability_treet::get_cur_state()
{
#ifdef DEBUG
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
#endif

  return **_cur_state_it;
}

const execution_statet & reachability_treet::get_cur_state() const
{

  return **_cur_state_it;
}

/*******************************************************************
 Function: reachability_treet::has_more_states

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::has_more_states()
{
#ifdef DEBUG
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
#endif

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
  if(_CS_bound  != -1 && get_cur_state().get_context_switch() >= _CS_bound)
  {
	_actual_CS_bound++;
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
  return _CS_bound;
}

/*******************************************************************
 Function: reachability_treet::get_actual_CS_bound

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

int reachability_treet::get_actual_CS_bound()
{
  return _actual_CS_bound;
}

/*******************************************************************
 Function: reachability_treet::is_global_assign

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::is_global_assign(const exprt &code)
{
#ifdef DEBUG
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
#endif

  int num_read_globals = get_cur_state().get_expr_read_globals(_ns,code.op1());

  if (get_is_same_mutex())
    return false;

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
#ifdef DEBUG
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
#endif

  if (get_is_same_mutex())
    return false;

  if (check_CS_bound())
    return false;

  if (get_cur_state().get_active_atomic_number() > 0)
   	return false;

  if (get_cur_state().get_expr_read_globals(_ns,code) > 0)
    return generate_states_base(code);
  else
    return false;
}

/*******************************************************************
 Function: reachability_treet::generate_states_before_write

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::generate_states_before_write(const exprt &code)
{
#ifdef DEBUG
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
#endif

  if (get_is_same_mutex())
    return false;

  if (check_CS_bound())
    return false;

  if (get_cur_state().get_active_atomic_number() > 0)
   	return false;

  if (get_cur_state().get_expr_write_globals(_ns, code) > 0)
    return generate_states_base(code);
  else
    return false;
}

/*******************************************************************
 Function: reachability_treet::get_is_mutex

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::get_is_same_mutex(void)
{
  return _is_same_mutex;
}

/*******************************************************************
 Function: reachability_treet::check_mutex

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void reachability_treet::check_mutex(const exprt &code, const execution_statet &ex_state)
{
#ifdef DEBUG
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
#endif

  static bool is_first_assign=true;
  static std::string identifier;
  const exprt &object=code.op0();
  const exprt &value=code.op1();
  std::string val;

  if (object.id() == "member")
  {
	if (object.op0().type().get_string("identifier").find("pthread_mutex") != std::string::npos)
	{
	  if (is_first_assign)
	  {
		if (object.op0().operands().size()==0)
		  return;
	    identifier = object.op0().op0().get_string("identifier");
	    is_first_assign=false;
	  }

	  val = integer2string(binary2integer(value.get_string("value"), true),10);

	  if (identifier.find(object.op0().op0().get_string("identifier")) != std::string::npos)
	    _is_same_mutex=true;
	  else if (val.find("0") == std::string::npos)
	    _is_same_mutex=false;

	  identifier = object.op0().op0().get_string("identifier");
    }
  }
}

/*******************************************************************
 Function: reachability_treet::generate_states_before_assign

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::generate_states_before_assign(const exprt &code, execution_statet &ex_state)
{
#ifdef DEBUG
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
#endif

  if(code.operands().size()!=2)
    throw "assignment expects two operands";

#if 0
  if (!_deadlock_detection)
    check_mutex(code, ex_state);

  if (get_is_same_mutex())
    return false;
#endif

  if(check_CS_bound())
    return false;

  if(get_cur_state().get_active_atomic_number() > 0)
    return false;

  int num_write_globals = get_cur_state().get_expr_write_globals(_ns,code.op0());
  int num_read_globals = get_cur_state().get_expr_read_globals(_ns,code.op1());

  //std::cout << "code.pretty(): " << code.pretty() << std::endl;
  //std::cout << "num_read_globals: " << num_read_globals << std::endl;
  //std::cout << "num_write_globals: " << num_write_globals << std::endl;

  if(num_read_globals + num_write_globals > 0)
  {
	ex_state.reexecute_instruction = false;
    return generate_states_base(code);
  }

  return false;
}

/*******************************************************************
 Function: reachability_treet::generate_states_before_function

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::generate_states_before_function(const code_function_callt &code)
{
#ifdef DEBUG
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
#endif

  if (get_is_same_mutex())
    return false;

  if(check_CS_bound())
    return false;

  if(get_cur_state().get_active_atomic_number() > 0)
   	return false;

  int num_globals = 0;

  for(std::vector<exprt>::const_iterator it=code.arguments().begin();
            it!=code.arguments().end(); it++)
  {
    num_globals += get_cur_state().get_expr_read_globals(_ns,*it);
  }
  if(num_globals > 0)
    return generate_states_base(code);

  return false;
}

/*******************************************************************
 Function: reachability_treet::generate_states_after_start_thread

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::generate_states_after_start_thread()
{
#ifdef DEBUG
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
#endif

  get_cur_state().reexecute_instruction = false;

  return generate_states_base(exprt());
}

/*******************************************************************
 Function: reachability_treet::generate_states

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::generate_states()
{
#ifdef DEBUG
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
#endif

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

  if (_por)
  {
    if(ex_state.last_global_expr.is_not_nil() && !expr.id().empty())
    {
      if(i < ex_state._active_thread)
      {
        if(ex_state.last_global_read_write.write_set.empty() &&
           ex_state._exprs_read_write.at(i+1).write_set.empty() &&
           ex_state._exprs_read_write.at(ex_state._active_thread).write_set.empty())
        {
          //std::cout << "empty: " << expr.pretty() << std::endl;
          //continue;
          return false;
        }

        consider = false;

        if(ex_state.last_global_read_write.has_write_intersect(ex_state._exprs_read_write.at(i+1).write_set))
        {
          consider = true;
          //std::cout << "write-write analysis" << std::endl;
        }
        else if(ex_state.last_global_read_write.has_write_intersect(ex_state._exprs_read_write.at(i+1).read_set))
        {
          consider = true;
          //std::cout << "write-read analysis" << std::endl;
        }
        else if(ex_state.last_global_read_write.has_read_intersect(ex_state._exprs_read_write.at(i+1).write_set))
        {
          consider = true;
          //std::cout << "read-write analysis" << std::endl;
        }

        //std::cout << "consider: " << consider << std::endl;
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
#ifdef DEBUG
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
  std::cout << expr.pretty() << std::endl;
#endif

//  std::cout << "generate_states_base expr.pretty(): " << expr.pretty() << std::endl;

  if(_CS_bound  != -1 && get_cur_state().get_context_switch() >= _CS_bound)
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

  if(ex_state._threads_state.size() < 2)
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

  unsigned int tid;

  for(tid = 0; tid < ex_state._threads_state.size(); tid++)
  {
    /* For all threads: */

    /* DFS -> depth first search? Check whether we've searched this... thing? */
    if(ex_state._DFS_traversed.at(tid))
      continue;

    ex_state._DFS_traversed.at(tid) = true;

    /* Presumably checks whether this thread isn't in user code yet? */
    if(ex_state._threads_state.at(tid).call_stack.empty())
      continue;

    /* Is it even still running? */
    if(ex_state._threads_state.at(tid).thread_ended)
      continue;

    //apply static partial-order reduction
    if(!apply_static_por(ex_state, expr, tid))
      continue;

    break;
  }

  _go_next = true;

  if (tid != ex_state._threads_state.size()) {

    /* Generate a new execution state, duplicate of previous? */
    execution_statet *new_state = new execution_statet(ex_state);
    execution_states.push_back(new_state);

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
 Function: reachability_treet::is_go_next_state

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::is_go_next_state()
{
#ifdef DEBUG
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
#endif

  return _go_next ||
         get_cur_state().get_active_state().thread_ended ||
         get_cur_state().get_active_state().call_stack.empty();
}

/*******************************************************************
 Function: reachability_treet::is_go_next_formula

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool reachability_treet::is_go_next_formula()
{
#ifdef DEBUG
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
#endif

  return _go_next_formula;
}

/*******************************************************************
 Function: reachability_treet::multi_formulae_go_next_state

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void reachability_treet::multi_formulae_go_next_state()
{
#ifdef DEBUG
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
#endif

  std::list<execution_statet*>::iterator it = _cur_state_it;
  it++;

  if(it != execution_states.end()) {
    _cur_state_it++;
  } else {
    if (generate_states_base(exprt())) {
      _cur_state_it++;
    } else {
      if (config.options.get_bool_option("print-stack-traces"))
        print_ileave_trace();
      _go_next_formula = true;
    }
  }

  _go_next = false;
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

  it = _cur_state_it--;
  delete *it;
  execution_states.erase(it);

  while(execution_states.size() > 0 && !generate_states_base(exprt())) {
    it = _cur_state_it--;
    delete *it;
    execution_states.erase(it);
  }

  if (execution_states.size() > 0)
    _cur_state_it++;

  _go_next = false;
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
#ifdef DEBUG
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
#endif

  std::list<execution_statet*>::iterator it = _cur_state_it;
  it++;
  if(it != execution_states.end())
    _cur_state_it++;
  else
  {
    while(execution_states.size() > 0 && !generate_states_base(exprt()))
    {
      it = _cur_state_it;
      _cur_state_it--;
      execution_states.erase(it);
    }

    if(execution_states.size() > 0)
      _cur_state_it++;
  }

  _go_next = false;
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
    state.num_threads = ex->_threads_state.size();
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

  f = fopen(filename.c_str(), "w");
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

  f = fopen(filename.c_str(), "r");
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
    (*it)->print_stack_traces(4);
  }
}

int
reachability_treet::get_ileave_direction_from_user(const exprt &expr) const
{
  std::list<execution_statet*>::const_iterator it;
  std::string input;
  unsigned int tid;

  std::cout << "Context switch point encountered; please select a thread to run" << std::endl;
  std::cout << "Current thread states:" << std::endl;
  for (it = execution_states.begin(); it != execution_states.end(); it++)
    (*it)->print_stack_traces(4);

  while (true) {
    std::cout << "Input: ";
    std::cin >> input;
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
      } else if (tid >= get_cur_state()._threads_state.size()) {
        std::cout << "Number out of range";
      } else {
        if (!check_thread_viable(tid, expr))
          break;
      }
    }
  }

  return tid;
}

bool
reachability_treet::check_thread_viable(int tid, const exprt &expr) const
{
  const execution_statet &ex = get_cur_state();

  if (ex._DFS_traversed.at(tid) == true) {
    std::cout << "Thread unschedulable as it's already been explored" << std::endl;
    return false;
  }

  if (ex._threads_state.at(tid).call_stack.empty()) {
    std::cout << "Thread unschedulable due to empty call stack" << std::endl;
    return false;
  }

  if (ex._threads_state.at(tid).thread_ended) {
    std::cout << "That thread has ended" << std::endl;
    return false;
  }

  if (!apply_static_por(ex, expr, tid)) {
    std::cout << "Thread unschedulable due to POR" << std::endl;
    return false;
  }

  return true;
}
