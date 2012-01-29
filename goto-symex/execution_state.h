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

class reachability_treet;

class execution_statet
{

public:
	execution_statet(const goto_functionst &goto_functions,
                const namespacet &ns, const reachability_treet *art,
                goto_symex_statet::level2t &l2, bool _is_schedule):
		owning_rt(art),
		_state_level2(l2),
                _target(ns),
                _goto_functions(goto_functions)
	{

          // XXXjmorse - C++s static initialization order trainwreck means
          // we can't initialize the id -> serializer map statically. Instead,
          // manually inspect and initialize. This is not thread safe.
          if (!execution_statet::expr_id_map_initialized) {
            execution_statet::expr_id_map_initialized = true;
            execution_statet::expr_id_map = init_expr_id_map();
          }

		is_schedule = _is_schedule;
		reexecute_instruction = true;
		reexecute_atomic = false;
		_CS_number = 0;
		_actual_CS_number=0;
		_TS_number = 0;
		node_id = 0;
		guard_execution = "execution_statet::\\guard_exec";
		guard_thread = "execution_statet::\\trdsel";

		goto_functionst::function_mapt::const_iterator it=
				goto_functions.function_map.find("main");
		if(it==goto_functions.function_map.end())
			throw "main symbol not found; please set an entry point";

		_goto_program =&(it->second.body);

                // Initialize initial thread state
                goto_symex_statet state(_state_level2);
                state.initialize((*_goto_program).instructions.begin(),
                                 (*_goto_program).instructions.end(),
                                 _goto_program, 0);
                _threads_state.push_back(state);
                _atomic_numbers.push_back(0);

                if (_DFS_traversed.size() <= state.source.thread_nr) {
                  _DFS_traversed.push_back(false);
                } else {
                  _DFS_traversed[state.source.thread_nr] = false;
                }

                _exprs.push_back(exprt());
                _exprs_read_write.push_back(read_write_set());

		_active_thread = 0;
		_last_active_thread = 0;
		generating_new_threads = 0;
		node_count=0;
		nondet_count = 0;
		dynamic_counter = 0;
		_DFS_traversed.reserve(1);
		_DFS_traversed[0] = false;

                next_thread_start_arg = exprt();
                next_thread_start_func = exprt();

		str_state = string_container.take_state_snapshot();
	};

	execution_statet(const execution_statet &ex) :
		owning_rt(ex.owning_rt),
		_state_level2(ex._state_level2),
		_target(ex._target),
		_goto_functions(ex._goto_functions)
	{
		*this = ex;

		// Don't copy string state in this copy constructor - instead
		// take another snapshot to represent what string state was
		// like when we began the exploration this execution_statet will
		// perform.
		str_state = string_container.take_state_snapshot();

		// Regenerate threads state using new objects _state_level2 ref
		_threads_state.clear();
		std::vector<goto_symex_statet>::const_iterator it;
		for(it = ex._threads_state.begin(); it != ex._threads_state.end(); it++) {
			goto_symex_statet state(*it, _state_level2);
			_threads_state.push_back(state);
		}

	}

	execution_statet& operator=(const execution_statet &ex)
	{
		is_schedule = ex.is_schedule;
		_threads_state = ex._threads_state;
		_atomic_numbers = ex._atomic_numbers;
		_DFS_traversed = ex._DFS_traversed;
		_exprs = ex._exprs;
		generating_new_threads = ex.generating_new_threads;
		last_global_expr = ex.last_global_expr;
		_exprs_read_write = ex._exprs_read_write;
		last_global_read_write = ex.last_global_read_write;
		_last_active_thread = ex._last_active_thread;
		_state_level2 = ex._state_level2;
		_active_thread = ex._active_thread;
		guard_execution = ex.guard_execution;
		guard_thread = ex.guard_thread;
		_parent_guard_identifier = ex._parent_guard_identifier;
		reexecute_instruction = ex.reexecute_instruction;
		reexecute_atomic = ex.reexecute_atomic;
		_actual_CS_number = ex._actual_CS_number;
		nondet_count = ex.nondet_count;
		dynamic_counter = ex.dynamic_counter;
		node_id = ex.node_id;
		parent_node_id = ex.parent_node_id;

		_goto_program = ex._goto_program;
		_CS_number = ex._CS_number;
		_TS_number = ex._TS_number;

                next_thread_start_arg = ex.next_thread_start_arg;
                next_thread_start_func = ex.next_thread_start_func;
		return *this;
	}

	virtual ~execution_statet() {
		// Free all name strings and suchlike we generated on this run
		// and no longer require
		// But, not if we're running with --schedule, as we'll need all
		// that information later.
		if (!is_schedule)
			string_container.restore_state_snapshot(str_state);
	};

    // Types

    typedef std::string (*serialise_fxn)(execution_statet &ex_state, const exprt &rhs);
    typedef std::map<const irep_idt, serialise_fxn> expr_id_map_t;

    // Methods

    /* number of context switches we've performed to reach this state */
    void increment_context_switch()
    {
      _CS_number++;
    }

    void increment_time_slice()
    {
      _TS_number++;
    }

    void reset_time_slice()
    {
      _TS_number=0;
    }

    int get_context_switch()
    {
      return _CS_number;
    }

    int get_time_slice()
    {
      return _TS_number;
    }

    void reset_DFS_traversed()
    {
   	  for(unsigned int i=0;i<_threads_state.size();i++)
		_DFS_traversed.at(i) = false;
    }

    void set_next_thread_start_arg(exprt &e)
    {
      next_thread_start_arg = e;
    }

    const exprt &get_next_thread_start_arg()
    {
      return next_thread_start_arg;
    }

    void set_next_thread_start_func(exprt &e)
    {
      next_thread_start_func = e;
    }

    const exprt &get_next_thread_start_func()
    {
      return next_thread_start_func;
    }

    void recover_global_state(const namespacet &ns, symex_targett &target);

    irep_idt get_guard_identifier();
    irep_idt get_guard_identifier_base();
    void set_parent_guard(const irep_idt & parent_guard);
    bool all_threads_ended();
	goto_symex_statet & get_active_state();
	const goto_symex_statet & get_active_state() const;
	unsigned int get_active_state_number() { return _active_thread; }
	unsigned int get_active_atomic_number();
	void increment_active_atomic_number();
	void decrement_active_atomic_number();
    void set_state_guard(unsigned int i, const exprt & guard);
	void set_active_state(unsigned int i);
    void execute_guard(const namespacet & ns, symex_targett &target);

    void add_thread(goto_programt *prog);
    void end_thread();
    /* Presumably this does the same as read_globals, see below */
    unsigned int get_expr_write_globals(const namespacet &ns, const exprt & expr);
    /* This takes the given expression, and for all constituent parts looks
     * through the identifiers that it touches and checks to see whether or
     * not they're globals. Counts them; also puts them in the
     * _exprs_read_write implicitly as reads: my eyes are on fire. */
    unsigned int get_expr_read_globals(const namespacet &ns, const exprt & expr);

    //void deadlock_detection(const namespacet &ns, symex_targett &target);
    void increament_trds_in_run(const namespacet &ns, symex_targett &target);
    void update_trds_count(const namespacet &ns, symex_targett &target);
    //void update_trds_status(const namespacet &ns, symex_targett &target);

    crypto_hash generate_hash(void) const;
    crypto_hash update_hash_for_assignment(const exprt &rhs);
    std::string serialise_expr(const exprt &rhs);

    void print_stack_traces(const namespacet &ns, unsigned int indent = 0) const;

private:
    void decreament_trds_in_run(const namespacet &ns, symex_targett &target);
    const symbolt& lookup(const namespacet &ns, const irep_idt &identifier)  const;
    bool is_in_lookup(const namespacet &ns, const irep_idt &identifier) const;

    // Object state

public:

    const reachability_treet *owning_rt;

    /* jmorse - Set of current thread states, indexed by threads id number*/
    std::vector<goto_symex_statet> _threads_state;

    /* jmorse - appears to just be a flag indicating whether we're currently
     * in an atomic section */
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
    goto_symex_statet::level2t _state_level2;
    unsigned int _active_thread;

    irep_idt guard_execution;
    irep_idt guard_thread;
    irep_idt _parent_guard_identifier;

    // Is the "--schedule" option enabled?
    bool is_schedule;

    bool reexecute_instruction; // temporarily disable context switch for the thread inherited from the last active thread
    bool reexecute_atomic; // temporarily disable context switch for the thread inherited from the last active thread
    int _actual_CS_number; //count the actual number of context switches

    int _TS_number;

    unsigned nondet_count;
    unsigned dynamic_counter;

    unsigned int node_id;
    unsigned int parent_node_id;
    symex_target_equationt _target;

private:
    const goto_functionst &_goto_functions;
    const goto_programt *_goto_program;
    int _CS_number;
    string_containert::str_snapshot str_state;

    exprt next_thread_start_arg;
    exprt next_thread_start_func;

    // Static stuff:

public:
    static expr_id_map_t init_expr_id_map();
    static bool expr_id_map_initialized;
    static expr_id_map_t expr_id_map;
    static unsigned int node_count;
};

#endif /* EXECUTION_STATE_H_ */
