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
#include "renaming.h"

class reachability_treet;

class execution_statet : public goto_symext
{
  public: class ex_state_level2t; // Forward dec

  public:
  execution_statet(const goto_functionst &goto_functions, const namespacet &ns,
                   reachability_treet *art,
                   symex_targett *_target,
                   contextt &context,
                   ex_state_level2t *l2init,
                   const optionst &options);

  execution_statet(const execution_statet &ex);
  execution_statet &operator=(const execution_statet &ex);
  virtual ~execution_statet();

  // Types

  typedef std::string (*serialise_fxn)(execution_statet &ex_state,
                                       const exprt &rhs);
  typedef std::map<const irep_idt, serialise_fxn> expr_id_map_t;

  class ex_state_level2t : public renaming::level2t
  {
  public:
    ex_state_level2t(execution_statet &ref);
    virtual ~ex_state_level2t();
    virtual ex_state_level2t *clone(void) const;
    virtual void rename(const irep_idt &identifier, unsigned count);

    execution_statet *owner;
  };

  class state_hashing_level2t : public ex_state_level2t
  {
  public:
    virtual state_hashing_level2t *clone(void) const;
    virtual irep_idt make_assignment(irep_idt l1_ident,
                                     const exprt &const_value,
                                     const exprt &assigned_value);
    typedef std::map<irep_idt, crypto_hash> current_state_hashest;
    current_state_hashest current_hashes;
  };

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

  virtual execution_statet *clone(void) const = 0;
  virtual void symex_step(const goto_functionst &goto_functions,
                          reachability_treet &art);
  virtual void symex_assign(statet &state, execution_statet &ex_state, const codet &code);
  virtual void claim(const exprt &expr, const std::string &msg, statet &state);
  virtual void symex_goto(statet &state, execution_statet &ex_state,
                          const exprt &old_guard);
  virtual void assume(const exprt &assumption, statet &state);

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
  bool dfs_explore_thread(unsigned int tid);
  bool check_if_ileaves_blocked(void);
  bool apply_static_por(const exprt &expr, int i) const;

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

  reachability_treet *owning_rt;
  std::vector<goto_symex_statet> threads_state;
  std::vector<unsigned int> atomic_numbers;
  std::vector<bool> DFS_traversed;
  std::vector<read_write_set> exprs_read_write;
  read_write_set last_global_read_write;
  unsigned int last_active_thread;
  ex_state_level2t *state_level2;
  unsigned int active_thread;
  irep_idt guard_execution;
  irep_idt parent_guard_identifier;
  int TS_number;
  unsigned nondet_count;
  unsigned dynamic_counter;
  unsigned int node_id;

  protected:
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

class dfs_execution_statet : public execution_statet
{
  public:
  dfs_execution_statet(
                   const goto_functionst &goto_functions,
                   const namespacet &ns,
                   reachability_treet *art,
                   symex_targett *_target,
                   contextt &context,
                   const optionst &options)
      : execution_statet(goto_functions, ns, art, _target, context,
                         new ex_state_level2t(*this), options)
  {
  };

  dfs_execution_statet(const dfs_execution_statet &ref);
  dfs_execution_statet *clone(void) const;
  virtual ~dfs_execution_statet(void);
};

class schedule_execution_statet : public execution_statet
{
  public:
  schedule_execution_statet(
                   const goto_functionst &goto_functions,
                   const namespacet &ns,
                   reachability_treet *art,
                   symex_targett *_target,
                   contextt &context,
                   const optionst &options,
                   unsigned int *ptotal_claims,
                   unsigned int *premaining_claims)
      : execution_statet(goto_functions, ns, art, _target, context,
                         new ex_state_level2t(*this), options)
  {
    this->ptotal_claims = ptotal_claims;
    this->premaining_claims = premaining_claims;
    *ptotal_claims = 0;
    *premaining_claims = 0;
  };

  schedule_execution_statet(const schedule_execution_statet &ref);
  schedule_execution_statet *clone(void) const;
  virtual ~schedule_execution_statet(void);
  virtual void claim(const exprt &expr, const std::string &msg, statet &state);

  unsigned int *ptotal_claims;
  unsigned int *premaining_claims;
};

#endif /* EXECUTION_STATE_H_ */
