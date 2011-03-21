/*******************************************************************\

Module:

Author: Ben YIU, yspb1g08@ecs.soton.ac.uk
		Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/


#ifndef EXECUTION_STATE_H_
#define EXECUTION_STATE_H_

#include <iostream>
#include <deque>
#include <set>
#include <map>
#include <list>
#include <algorithm>
#include <std_expr.h>
#include "symex_target_equation.h"

#include "goto_symex_state.h"
#include "read_write_set.h"

class execution_statet
{

public:
	execution_statet(const goto_functionst &goto_functions,
                const namespacet &ns):
                _target(ns),
                _goto_functions(goto_functions)
	{
	  reexecute_instruction = true;
	  reexecute_atomic = false;
      _CS_number = 0;
      _actual_CS_number=0;
      node_id = 0;
      guard_execution = "execution_statet::\\guard_exec";
      guard_thread = "execution_statet::\\trdsel";

	  goto_functionst::function_mapt::const_iterator it=
		    goto_functions.function_map.find("main");
	  if(it==goto_functions.function_map.end())
		    throw "main symbol not found; please set an entry point";

	  _goto_program =&(it->second.body);

	  _state_level2 = new goto_symex_statet::level2t();
	  add_thread((*_goto_program).instructions.begin(),(*_goto_program).instructions.end());
	  _active_thread = 0;
      _last_active_thread = 0;
      generating_new_threads = 0;
	  node_count=0;
	};

	virtual ~execution_statet()	{};

    /* number of context switches we've performed to reach this state */
    void increment_context_switch()
    {
      _CS_number++;
    }

    int get_context_switch()
    {
      return _CS_number;
    }

	void free_level2()
	{
      delete _state_level2;
	}

	void copy_level2_from(execution_statet & ex_state)
	{
	  _state_level2 = new goto_symex_statet::level2t(*ex_state._state_level2);
	  for(unsigned int i=0;i<_threads_state.size();i++)
	    _threads_state.at(i).level2 = _state_level2;
	}

    void reset_DFS_traversed()
    {
   	  for(unsigned int i=0;i<_threads_state.size();i++)
		_DFS_traversed.at(i) = false;
    }

    static unsigned int node_count;
    unsigned int node_id;
    unsigned int parent_node_id;
    symex_target_equationt _target;

    irep_idt get_guard_identifier();
    irep_idt get_guard_identifier_base();
    void set_parent_guard(const irep_idt & parent_guard);
    bool all_threads_ended();
	goto_symex_statet & get_active_state();
	unsigned int get_active_atomic_number();
	void increment_active_atomic_number();
	void decrement_active_atomic_number();
    void set_state_guard(unsigned int i, const exprt & guard);
	void set_active_state(unsigned int i);
    void execute_guard(const namespacet & ns, symex_targett &target);

	void add_thread(goto_programt::const_targett start, goto_programt::const_targett end);
	void add_thread(goto_symex_statet & state);
    void end_thread(const namespacet &ns, symex_targett &target);
    /* Presumably this does the same as read_globals, see below */
    unsigned int get_expr_write_globals(const namespacet &ns, const exprt & expr);
    /* This takes the given expression, and for all constituent parts looks
     * through the identifiers that it touches and checks to see whether or
     * not they're globals. Counts them; also puts them in the
     * _exprs_read_write implicitly as reads: my eyes are on fire. */
    unsigned int get_expr_read_globals(const namespacet &ns, const exprt & expr);

	/* jmorse - Set of current thread states, indexed by threads id number*/
	std::vector<goto_symex_statet> _threads_state;
	std::vector<unsigned int> _atomic_numbers;
	/* jmorse - Depth first search? */
	std::vector<bool> _DFS_traversed;
	/* jmorse - a set of expressions, one for each active thread, showing
	 * where each thread is at? generate_states_base. */
	std::vector<exprt> _exprs;
    int generating_new_threads;
    /* jmorse - Presumably the last expr to be executed */
    exprt last_global_expr;
    /* jmorse - a set of operations (irep_idts; identifiers?) that presumably
     * occur at the top of each state. indexed by thread id no. So, it's the
     * set of most recent reads/writes of thread? */
    std::vector<read_write_set> _exprs_read_write;
    /* jmorse - what the name says */
    read_write_set last_global_read_write;

    unsigned int _last_active_thread;
	goto_symex_statet::level2t *_state_level2;
	unsigned int _active_thread;

    irep_idt guard_execution;
    irep_idt guard_thread;
    irep_idt _parent_guard_identifier;

    bool reexecute_instruction; // temporarily disable context switch for the thread inherited from the last active thread
    bool reexecute_atomic; // temporarily disable context switch for the thread inherited from the last active thread
    int _actual_CS_number; //count the actual number of context switches

    //void deadlock_detection(const namespacet &ns, symex_targett &target);
    void increament_trds_in_run(const namespacet &ns, symex_targett &target);
    void update_trds_count(const namespacet &ns, symex_targett &target);
    void set_trd_status_to_one(const namespacet &ns, symex_targett &target);
    void set_trd_status_to_zero(const namespacet &ns, symex_targett &target);
    void set_active_trd_one(const namespacet &ns, symex_targett &target);
    void set_active_trd_zero(const namespacet &ns, symex_targett &target);
    void update_trds_status(const namespacet &ns, symex_targett &target);

private:
    void decreament_trds_in_run(const namespacet &ns, symex_targett &target);
    const symbolt& lookup(const namespacet &ns, const irep_idt &identifier)  const;
    bool is_in_lookup(const namespacet &ns, const irep_idt &identifier) const;
	const goto_functionst &_goto_functions;
    const goto_programt *_goto_program;
    int _CS_number;
};

#endif /* EXECUTION_STATE_H_ */
