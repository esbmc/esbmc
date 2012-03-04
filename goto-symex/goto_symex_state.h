/*******************************************************************\

   Module: Symbolic Execution

   Author: Daniel Kroening, kroening@kroening.com Lucas Cordeiro,
     lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#ifndef CPROVER_GOTO_SYMEX_GOTO_SYMEX_STATE_H
#define CPROVER_GOTO_SYMEX_GOTO_SYMEX_STATE_H

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <guard.h>
#include <pointer-analysis/value_set.h>
#include <goto-programs/goto_functions.h>
#include <string>
#include <stack>
#include <vector>

#include "symex_target.h"
#include "crypto_hash.h"
#include "renaming.h"

#include <i2string.h>

class execution_statet; // foward dec

// central data structure: state
class goto_symex_statet
{
public:
  class goto_statet; // forward dec
  class framet; // forward dec

  goto_symex_statet(renaming::level2t &l2, value_sett &vs);
  goto_symex_statet(const goto_symex_statet &state, renaming::level2t &l2);

  goto_symex_statet &
  operator=(const goto_symex_statet &state);

  // Types

  typedef std::list<goto_statet> goto_state_listt;
  typedef std::map<goto_programt::const_targett,
                   goto_state_listt> goto_state_mapt;
  typedef std::vector<framet> call_stackt;
  typedef std::set<std::string> declaration_historyt;

  class goto_statet
  {
  public:
    unsigned depth;
    renaming::level2t *level2_ptr;
    renaming::level2t &level2;
    value_sett value_set;
    guardt guard;
    unsigned int thread_id;

    explicit
    goto_statet(const goto_symex_statet &s) :
      depth(s.depth),
      level2_ptr(s.level2.clone()),
      level2(*level2_ptr),
      value_set(s.value_set),
      guard(s.guard),
      thread_id(s.source.thread_nr)
    {
    }

    goto_statet(const goto_statet &s) :
      depth(s.depth),
      level2_ptr(s.level2_ptr->clone()),
      level2(*level2_ptr),
      value_set(s.value_set),
      guard(s.guard),
      thread_id(s.thread_id) {}

  // Deny the use of goto_statet copy constructors
  private:
  goto_statet &operator=(const goto_statet &ref __attribute__((unused)))
  {
    assert(0);
  }

  public:
    ~goto_statet() {
      delete level2_ptr;
      return;
    }
  };

  // function calls
  class framet
  {
  public:
    irep_idt function_identifier;
    goto_state_mapt goto_state_map;
    renaming::level1t level1;
    symex_targett::sourcet calling_location;

    goto_programt::const_targett end_of_function;
    exprt return_value;

    typedef std::set<irep_idt> local_variablest;
    local_variablest local_variables;

    // Records containing data for dereferencing and running a function pointer.
    // Should only be nonzero sized when in the middle of running such a func
    // ptr.
    // Program target instruction, and the symbol of the func its in.
    std::list<std::pair<goto_programt::const_targett,exprt> >
      cur_function_ptr_targets;
    goto_programt::const_targett function_ptr_call_loc;
    goto_programt::const_targett function_ptr_combine_target;
    const code_function_callt *orig_func_ptr_call;

    framet(unsigned int thread_id) :
      return_value(static_cast<const exprt &>(get_nil_irep()))
    {
      level1._thread_id = thread_id;
    }
  };

  // Macros

  // does both levels of renaming
  std::string
  current_name(const irep_idt &identifier) const
  {
    return current_name(level2, identifier);
  }

  std::string
  current_name(
    const renaming::level2t &plevel2, const irep_idt &identifier) const
  {
    irep_idt temp = top().level1.get_ident_name(identifier);
    return plevel2.get_ident_name(temp);
  }

  std::string
  current_name(
    const goto_statet &goto_state, const irep_idt &identifier) const
  {
    return current_name(goto_state.level2, identifier);
  }

  inline framet &
  top()
  {
    assert(!call_stack.empty());
    return call_stack.back();
  }

  inline const framet &
  top() const
  {
    assert(!call_stack.empty());
    return call_stack.back();
  }

  inline framet &
  new_frame(unsigned int thread_id) {
    call_stack.push_back(framet(thread_id));
    return call_stack.back();
  }
  inline void
  pop_frame() {
    call_stack.pop_back();
  }
  inline const framet &
  previous_frame() {
    return *(--(--call_stack.end()));
  }

  // Methods

  void initialize(const goto_programt::const_targett & start,
                  const goto_programt::const_targett & end,
                  const goto_programt *prog,
                  unsigned int thread_id);

  void rename(exprt &expr, const namespacet &ns);
  void rename_address(exprt &expr, const namespacet &ns);
  void rename(typet &type, const namespacet &ns);

  void assignment(
    exprt &lhs, const exprt &rhs, const namespacet &ns, bool record_value);

  // what to propagate
  bool constant_propagation(const exprt &expr) const;
  bool constant_propagation_reference(const exprt &expr) const;

  // undoes both levels of renaming
  const irep_idt get_original_name(const irep_idt &identifier) const;
  void get_original_name(exprt &expr) const;

  void print_stack_trace(const namespacet &ns, unsigned int indent) const;
  std::vector<dstring> gen_stack_trace(void) const;

  // Members

  // distance from entry
  unsigned depth;

  bool sleeping;
  bool waiting;
  unsigned int join_count;
  bool thread_ended;

  guardt guard;
  symex_targett::sourcet source;
  std::map<irep_idt, unsigned> function_frame;
  std::map<symex_targett::sourcet, unsigned> unwind_map;
  std::map<irep_idt, unsigned> function_unwind;

  // we remember all declarations
  declaration_historyt declaration_history;

  bool use_value_set;
  renaming::level2t &level2;
  value_sett &value_set;

  call_stackt call_stack;
};

#endif
