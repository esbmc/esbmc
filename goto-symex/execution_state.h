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

#include "symex_target.h"
#include "goto_symex_state.h"
#include "goto_symex.h"
#include "read_write_set.h"

class reachability_treet;

class execution_statet : public goto_symext
{

  public:
  execution_statet(const goto_functionst &goto_functions, const namespacet &ns,
                   const reachability_treet *art,
                   symex_targett *_target,
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

  // Macros

  void
  increment_context_switch()
  {
    CS_number++;
  }

  void
  increment_time_slice()
  {
    TS_number++;
  }

  void
  reset_time_slice()
  {
    TS_number = 0;
  }

  int
  get_context_switch()
  {
    return CS_number;
  }

  int
  get_time_slice()
  {
    return TS_number;
  }

  void
  resetDFS_traversed()
  {
    for (unsigned int i = 0; i < threads_state.size(); i++)
      DFS_traversed.at(i) = false;
  }

  unsigned int
  get_active_state_number() {
    return active_thread;
  }

  // Methods

  irep_idt get_guard_identifier();
  irep_idt get_guard_identifier_base();
  void set_parent_guard(const irep_idt & parent_guard);
  bool all_threads_ended();
  goto_symex_statet & get_active_state();
  const goto_symex_statet & get_active_state() const;
  unsigned int get_active_atomic_number();
  void increment_active_atomic_number();
  void decrement_active_atomic_number();
  void set_active_state(unsigned int i);
  void execute_guard(const namespacet & ns);

  void add_thread(goto_programt::const_targett start,
                  goto_programt::const_targett end,
                  const goto_programt *prog);
  void add_thread(goto_symex_statet & state);
  void end_thread(void);
  unsigned int get_expr_write_globals(const namespacet &ns, const exprt & expr);
  unsigned int get_expr_read_globals(const namespacet &ns, const exprt & expr);

  void increment_trds_in_run(void);
  void update_trds_count(void);

  crypto_hash generate_hash(void) const;
  crypto_hash update_hash_for_assignment(const exprt &rhs);
  std::string serialise_expr(const exprt &rhs);

  void print_stack_traces(const namespacet &ns, unsigned int indent = 0) const;

  private:
  void decrement_trds_in_run(void);

  // Object state

  public:

  const reachability_treet *owning_rt;
  std::vector<goto_symex_statet> threads_state;
  std::vector<unsigned int> atomic_numbers;
  std::vector<bool> DFS_traversed;
  int generating_new_threads;
  std::vector<read_write_set> exprs_read_write;
  read_write_set last_global_read_write;
  unsigned int lastactive_thread;
  goto_symex_statet::level2t state_level2;
  unsigned int active_thread;
  irep_idt guard_execution;
  irep_idt guard_thread;
  irep_idt parent_guard_identifier;
  bool is_schedule;
  bool reexecute_instruction;
  int TS_number;
  unsigned nondet_count;
  unsigned dynamic_counter;
  unsigned int node_id;
  unsigned int parent_node_id;

  private:
  const goto_functionst &_goto_functions;
  int CS_number;
  string_containert::str_snapshot str_state;

  // Static stuff:

  public:
  static expr_id_map_t init_expr_id_map();
  static bool expr_id_map_initialized;
  static expr_id_map_t expr_id_map;
  static unsigned int node_count;
};

#endif /* EXECUTION_STATE_H_ */
