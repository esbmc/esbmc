/*******************************************************************\

   Module:

   Author: Ben YIU, yspb1g08@ecs.soton.ac.uk Lucas Cordeiro,
     lcc08r@ecs.soton.ac.uk

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
#include "goto_symex.h"
#include "read_write_set.h"

class reachability_treet;

class execution_statet : public goto_symext
{

  public:
  execution_statet(const goto_functionst &goto_functions, const namespacet &ns,
                   const reachability_treet *art,
                   goto_symex_statet::level2t &l2,
                   contextt &context,
                   const optionst &options,
                   bool _is_schedule);

  execution_statet(const execution_statet &ex);

  execution_statet &operator=(const execution_statet &ex);

  virtual ~execution_statet();

  // Types

  typedef std::string (*serialise_fxn)(execution_statet &ex_state,
                                       const exprt &rhs);
  typedef std::map<const irep_idt, serialise_fxn> expr_id_map_t;

  // Methods

  /* number of context switches we've performed to reach this state */
  void
  increment_context_switch()
  {
    _CS_number++;
  }

  void
  increment_time_slice()
  {
    _TS_number++;
  }

  void
  reset_time_slice()
  {
    _TS_number = 0;
  }

  int
  get_context_switch()
  {
    return _CS_number;
  }

  int
  get_time_slice()
  {
    return _TS_number;
  }

  void
  reset_DFS_traversed()
  {
    for (unsigned int i = 0; i < _threads_state.size(); i++)
      _DFS_traversed.at(i) = false;
  }

  void recover_global_state(const namespacet &ns, symex_targett &target);

  irep_idt get_guard_identifier();
  irep_idt get_guard_identifier_base();
  void set_parent_guard(const irep_idt & parent_guard);
  bool all_threads_ended();
  goto_symex_statet & get_active_state();
  const goto_symex_statet & get_active_state() const;
  unsigned int
  get_active_state_number() {
    return _active_thread;
  }
  unsigned int get_active_atomic_number();
  void increment_active_atomic_number();
  void decrement_active_atomic_number();
  void set_state_guard(unsigned int i, const exprt & guard);
  void set_active_state(unsigned int i);
  void execute_guard(const namespacet & ns);

  void add_thread(goto_programt::const_targett start,
                  goto_programt::const_targett end,
                  const goto_programt *prog);
  void add_thread(goto_symex_statet & state);
  void end_thread(const namespacet &ns, symex_targett &target);
  unsigned int get_expr_write_globals(const namespacet &ns, const exprt & expr);
  unsigned int get_expr_read_globals(const namespacet &ns, const exprt & expr);

  void increament_trds_in_run(const namespacet &ns, symex_targett &target);
  void update_trds_count(const namespacet &ns, symex_targett &target);

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
  std::vector<goto_symex_statet> _threads_state;
  std::vector<unsigned int> _atomic_numbers;
  std::vector<bool> _DFS_traversed;
  std::vector<exprt> _exprs;
  int generating_new_threads;
  exprt last_global_expr;
  std::vector<read_write_set> _exprs_read_write;
  read_write_set last_global_read_write;
  unsigned int _last_active_thread;
  goto_symex_statet::level2t _state_level2;
  unsigned int _active_thread;
  irep_idt guard_execution;
  irep_idt guard_thread;
  irep_idt _parent_guard_identifier;
  bool is_schedule;
  bool reexecute_instruction;
  bool reexecute_atomic;
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

  // Static stuff:

  public:
  static expr_id_map_t init_expr_id_map();
  static bool expr_id_map_initialized;
  static expr_id_map_t expr_id_map;
  static unsigned int node_count;
};

#endif /* EXECUTION_STATE_H_ */
