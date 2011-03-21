/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/


#include "reachability_tree.h"
#include <i2string.h>
#include <expr_util.h>
#include <std_expr.h>

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

  return *_cur_state_it;
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

  if (!_deadlock_detection)
    check_mutex(code, ex_state);

  if (get_is_same_mutex())
    return false;

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

bool reachability_treet::apply_static_por(execution_statet &ex_state, const exprt &expr, int i)
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

  //new
#if 1
  if(expr.is_not_nil())
  {
    ex_state.last_global_expr = ex_state._exprs.at(ex_state._active_thread);
    ex_state.last_global_read_write = ex_state._exprs_read_write.at(ex_state._active_thread);
  }
#endif

  bool generated = false;

  for(unsigned int i = 0; i < ex_state._threads_state.size(); i++)
  {
    /* For all threads: */

    /* DFS -> depth first search? Check whether we've searched this... thing? */
    if(ex_state._DFS_traversed.at(i))
      continue;

    ex_state._DFS_traversed.at(i) = true;

    /* Presumably checks whether this thread isn't in user code yet? */
    if(ex_state._threads_state.at(i).call_stack.empty())
      continue;

    /* Is it even still running? */
    if(ex_state._threads_state.at(i).thread_ended)
      continue;

    //apply static partial-order reduction
    if(!apply_static_por(ex_state, expr, i))
      continue;

    /* Generate a new execution state, duplicate of previous? */
    execution_statet new_state(ex_state);
    execution_states.push_back(new_state);

    /* Make it active, make it follow on from previous state... */
    execution_states.rbegin()->set_active_state(i);
    execution_states.rbegin()->set_parent_guard(ex_state.get_guard_identifier());
    execution_states.rbegin()->reexecute_instruction = true;

    execution_states.rbegin()->increment_context_switch();
    /* ^^^ What if there /wasn't/ a switch though? */
    execution_states.rbegin()->copy_level2_from(ex_state);
    /* Reset interleavings (?) investigated in this new state */
    execution_states.rbegin()->reset_DFS_traversed();

    goto_programt::const_targett pc = execution_states.rbegin()->get_active_state().source.pc;
    pc_hits[*pc]++;
    pc_hit_iters[*pc] = pc;

    generated = true;
    break;

  }

  _go_next = true;

  return generated;

  return true;
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

  std::list<execution_statet>::iterator it = _cur_state_it;
  it++;

  if(it != execution_states.end())
    _cur_state_it++;
  else
  {
    bool last_state = true;
    while(execution_states.size() > 0 && !generate_states_base(exprt()))
    {
      it = _cur_state_it;
      _cur_state_it--;
      _go_next_formula = true;
      if(last_state)
      {
        if(_cur_target_state != NULL)
        delete _cur_target_state;
        _cur_target_state = new execution_statet(*it);
        last_state = false;
      }
      execution_states.erase(it);
    }

    if(execution_states.size() > 0)
      _cur_state_it++;
 }

	_go_next = false;
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

  std::list<execution_statet>::iterator it = _cur_state_it;
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

void reachability_treet::print_hits()
{
  std::map<goto_programt::instructiont, int>::const_iterator it;

  for (it = pc_hits.begin(); it != pc_hits.end(); it++) {
    std::cout << "Location " << (*it).first.location.as_string() << " hit " << (*it).second << " times\n";
  }

  return;
}
