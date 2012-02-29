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
    contextt &context);

  virtual ~reachability_treet() { };

  execution_statet & get_cur_state();
  const execution_statet & get_cur_state() const;
  bool reset_to_unexplored_state();
  bool has_more_states();
  int get_CS_bound() const;
  int get_ileave_direction_from_user(const exprt &expr) const;
  int get_ileave_direction_from_scheduling(const exprt &expr) const;
  bool check_thread_viable(int tid, const exprt &expr, bool quiet) const;
  bool analyse_for_cswitch_base(const exprt & expr);
  bool force_cswitch_point();

  bool analyse_for_cswitch_after_read(const exprt &code);
  bool analyse_for_cswitch_after_assign(const exprt &code);
  void create_next_state(void);
  bool step_next_state(void);
  bool is_global_assign(const exprt &code);

  unsigned int decide_ileave_direction(execution_statet &ex_state,
                                       const exprt &expr);

  void print_ileave_trace(void) const;
  bool is_at_end_of_run();
  bool is_has_complete_formula();
  void go_next_state();
  void switch_to_next_execution_state();

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
  int TS_slice;
  unsigned int schedule_total_claims, schedule_remaining_claims;
  unsigned int next_thread_id;
  bool deadlock_detection, por;
  const namespacet &ns;

    /* jmorse */
  std::set<crypto_hash>hit_hashes;

  optionst options;
};

#endif /* REACHABILITY_TREE_H_ */
