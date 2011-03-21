/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#ifndef REACHABILITY_TREE_H_
#define REACHABILITY_TREE_H_

#include <iostream>
#include <deque>
#include <set>
#include <map>
#include "execution_state.h"
#include "basic_symex.h"

#include "crypto_hash.h"

#include <goto-programs/goto_program.h>

class reachability_treet
{
public:
  reachability_treet(
    const goto_functionst &goto_functions,
    const namespacet &ns,
    int CS_bound,
    bool deadlock_detection,
    bool por):
    _goto_functions(goto_functions),
    _cur_target_state(NULL),
    _ns(ns)
	{
      _DFS=true;
	  _go_next = false;
	  _go_next_formula = false;
      _CS_bound = CS_bound;
      _actual_CS_bound = CS_bound;
      _deadlock_detection = deadlock_detection;
      _por = por;
      _is_same_mutex=false;
      execution_statet ex_state(goto_functions, ns);
	  execution_states.push_back(ex_state);
      _cur_state_it = execution_states.begin();
    };

	virtual ~reachability_treet() {};

	execution_statet & get_cur_state();
	bool has_more_states();
	bool check_CS_bound();
	int get_CS_bound();
	int get_actual_CS_bound();
	bool generate_states_base(const exprt & expr);
	bool apply_static_por(execution_statet &ex_state, const exprt &expr, int i);
    bool generate_states_after_start_thread();
    bool generate_states();

    bool get_is_same_mutex(void);
    void check_mutex(const exprt &code, const execution_statet &ex_state);

    bool generate_states_before_read(const exprt &code);
    bool generate_states_before_write(const exprt &code);
    bool generate_states_before_assign(const exprt &code, execution_statet &ex_state);
    bool generate_states_before_function(const code_function_callt &code);
    bool is_global_assign(const exprt &code);

    const symbolt &lookup(const namespacet &ns, const irep_idt &identifier) const;
    unsigned int get_write_globals(const namespacet &ns, const exprt & expr);
    unsigned int get_expr_read_globals(const namespacet &ns, const exprt & expr);

	bool is_go_next_state();
	bool is_go_next_formula();
	void go_next_state();
	void multi_formulae_go_next_state();
    void set_go_next_state()
    {
      _go_next = true;
    }

	const goto_functionst &_goto_functions;
    execution_statet* _cur_target_state;
	bool _go_next_formula;
private:
	std::list<execution_statet> execution_states;
    /* This is derefed and returned by get_current_state */
    std::list<execution_statet>::iterator _cur_state_it;
    int _CS_bound, _actual_CS_bound;
	bool _go_next;
	bool _DFS, _multi_formulae, _is_same_mutex,
		 _deadlock_detection, _por;
    const namespacet &_ns;

    /* jmorse */
    std::set<crypto_hash>hit_hashes;
};

#endif /* REACHABILITY_TREE_H_ */
