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
#include <options.h>
#include "goto_symex.h"
#include "execution_state.h"
#include "symex_target_equation.h"
#include "renaming.h"

// Can't include goto_symex.h due to inclusion order. This can be fixed with the
// refactor; in the meantime, forward dec.

#include "crypto_hash.h"

#include <goto-programs/goto_program.h>

class reachability_treet
{
public:
  reachability_treet(
    const goto_functionst &goto_functions,
    const namespacet &ns,
    const optionst &opts,
    symex_targett *target,
    contextt &context) :
    goto_functions(goto_functions),
    reached_terminal_state(NULL),
    ns(ns),
    options(opts)
  {
    CS_bound = atoi(options.get_option("context-switch").c_str());
    deadlock_detection = options.get_bool_option("deadlock-check");
    _TS_slice = atoi(options.get_option("time-slice").c_str());
    state_hashing = options.get_bool_option("state-hashing");
    directed_interleavings = options.get_bool_option("direct-interleavings");

    if (options.get_bool_option("no-por") || options.get_bool_option("control-flow-test"))
      por = false;
    else
      por = true;

    at_end_of_run = false;
    has_complete_formula = false;
    is_same_mutex=false;

    execution_statet *s;
    if (options.get_bool_option("schedule")) {
      s = reinterpret_cast<execution_statet*>(
                           new schedule_execution_statet(goto_functions, ns,
                                                 this, target, context, opts,
                                                 &schedule_total_claims,
                                                 &schedule_remaining_claims));
      schedule_target = target;
    } else {
      s = reinterpret_cast<execution_statet*>(
                           new dfs_execution_statet(goto_functions, ns, this,
                                                 target, context, opts));
      schedule_target = NULL;
    }

    execution_states.push_back(s);
    cur_state_it = execution_states.begin();
  };

  virtual ~reachability_treet() { };

  execution_statet & get_cur_state();
  const execution_statet & get_cur_state() const;
  bool reset_to_unexplored_state();
  bool has_more_states();
  bool check_CS_bound();
  int get_CS_bound() const;
  int get_actual_CS_bound();
  int get_ileave_direction_from_user(const exprt &expr) const;
  int get_ileave_direction_from_scheduling(const exprt &expr) const;
  bool check_thread_viable(int tid, const exprt &expr, bool quiet) const;
  bool generate_states_base(const exprt & expr);
  bool apply_static_por(const execution_statet &ex_state, const exprt &expr, int i) const;
  bool generate_states();

  bool generate_states_before_read(const exprt &code);
  bool generate_states_before_assign(const exprt &code, execution_statet &ex_state);
  bool is_global_assign(const exprt &code);

  void print_ileave_trace(void) const;
  bool is_at_end_of_run();
  bool is_has_complete_formula();
  void go_next_state();
  void switch_to_next_execution_state();
  void set_is_at_end_of_run()
  {
    at_end_of_run = true;
  }

  // Interface for bmc operation goes here

  goto_symext::symex_resultt *get_next_formula();
  goto_symext::symex_resultt *generate_schedule_formula();
  bool setup_next_formula(void);

  class dfs_position {
public:
    dfs_position(const reachability_treet &rt);
    dfs_position(const std::string filename);
    bool write_to_file(const std::string filename) const;
protected:
    bool read_from_file(const std::string filename);
public:
    struct dfs_state {
      unsigned int location_number;
      unsigned int num_threads;
      unsigned int cur_thread;
      std::vector<bool> explored;
    };

    static const uint32_t file_magic;

    struct file_hdr {
      uint32_t magic;
      uint32_t checksum;
      uint32_t num_states;
      uint32_t num_ileaves;
    };

    struct file_entry {
      uint32_t location_number;
      uint16_t num_threads;
      uint16_t cur_thread;
      // Followed by bitfield for threads explored state.
    };

    std::vector<struct dfs_state> states;

    // Number of interleavings explored to date.
    unsigned int ileaves;

    // We need to be able to detect when the source files have changed somehow,
    // leading to the checkpoint being invalid. So add a checksum field. Exactly
    // how it's going to be calculated, I don't know yet.
    uint64_t checksum;
  };

  bool restore_from_dfs_state(void *dfs);
  void save_checkpoint(const std::string fname) const;

  const goto_functionst &goto_functions;

  // The current terminating execution state that we've reached
  execution_statet* reached_terminal_state;
  // Has complete formula: we have executed up to the end of the program and
  // we have an SSA formula we can verify. When this occurs, we drop back and
  // let the higher level code convert the formula/equation.
  bool has_complete_formula;
  // End of run: where we have executed up to the point where there is a
  // context switch that may be taken.
  bool at_end_of_run;
  bool state_hashing;
  bool directed_interleavings;
protected:
  std::list<execution_statet*> execution_states;
  /* This is derefed and returned by get_current_state */
  std::list<execution_statet*>::iterator cur_state_it;
  symex_targett *schedule_target;
  int CS_bound;
  int _TS_slice;
  unsigned int schedule_total_claims, schedule_remaining_claims;
  bool is_same_mutex, deadlock_detection, por;
  const namespacet &ns;

    /* jmorse */
  std::set<crypto_hash>hit_hashes;

  optionst options;
};

#endif /* REACHABILITY_TREE_H_ */
