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
  goto_symex_statet(renaming::level2t &l2)
    : level2(l2)
  {
    use_value_set = true;
    depth = 0;
    sleeping = false;
    waiting = false;
    join_count = 0;
    thread_ended = false;
  }

  goto_symex_statet(const goto_symex_statet &state, renaming::level2t &l2)
    : level2(l2)
  {
    *this = state;
  }

  goto_symex_statet &
  operator=(const goto_symex_statet &state)
  {
    depth = state.depth;
    sleeping = state.sleeping;
    waiting = state.waiting;
    waiting = state.waiting;
    join_count = state.join_count;
    thread_ended = state.thread_ended;
    guard = state.guard;
    source = state.source;
    function_frame = state.function_frame;
    unwind_map = state.unwind_map;
    function_unwind = state.function_unwind;
    declaration_history = state.declaration_history;
    use_value_set = state.use_value_set;
    value_set = state.value_set;
    call_stack = state.call_stack;
    return *this;
  }

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

  // we have a two-level renaming

  typedef std::set<std::string> declaration_historyt;

  // we remember all declarations
  declaration_historyt declaration_history;

  renaming::level2t &level2;

  void initialize(const goto_programt::const_targett & start,
                  const goto_programt::const_targett & end,
                  const goto_programt *prog,
                  unsigned int thread_id);

  void rename(exprt &expr, const namespacet &ns, unsigned node_id);
  void rename_address(exprt &expr, const namespacet &ns, unsigned node_id);
  void rename(typet &type, const namespacet &ns, unsigned node_id);

  void assignment(
    exprt &lhs, const exprt &rhs, const namespacet &ns, bool record_value,
    execution_statet &ex_state, unsigned exec_node_id);

  // what to propagate
  bool constant_propagation(const exprt &expr) const;
  bool constant_propagation_reference(const exprt &expr) const;

  // undoes both levels of renaming
  const irep_idt &get_original_name(const irep_idt &identifier) const;
  void get_original_name(exprt &expr) const;

  // does both levels of renaming
  std::string
  current_name(const irep_idt &identifier, unsigned node_id) const
  {
    return current_name(level2, identifier, node_id);
  }

  std::string
  current_name(
    const renaming::level2t &plevel2, const irep_idt &identifier,
    unsigned node_id) const
  {
    irep_idt temp = top().level1.get_ident_name(identifier, node_id);
    return plevel2.stupid_operator(temp, node_id);
  }

  bool use_value_set;

  // uses level 1 names
  value_sett value_set;

  class goto_statet
  {
  public:
    unsigned depth;
    renaming::level2t level2;
    value_sett value_set;
    guardt guard;
    unsigned int thread_id;

    explicit
    goto_statet(const goto_symex_statet &s) :
      depth(s.depth),
      level2(s.level2),
      value_set(s.value_set),
      guard(s.guard),
      thread_id(s.source.thread_nr)
    {
    }
  };

  std::string
  current_name(
    const goto_statet &goto_state, const irep_idt &identifier,
    unsigned node_id) const
  {
    return current_name(goto_state.level2, identifier, node_id);
  }

  // gotos
  typedef std::list<goto_statet> goto_state_listt;
  typedef std::map<goto_programt::const_targett,
                   goto_state_listt> goto_state_mapt;

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

    framet(unsigned int thread_id) :
      return_value(static_cast<const exprt &>(get_nil_irep()))
    {
    	level1._thread_id = thread_id;
    }
  };

  typedef std::vector<framet> call_stackt;
  call_stackt call_stack;

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

  void print_stack_trace(const namespacet &ns, unsigned int indent) const;
  std::vector<dstring> gen_stack_trace(void) const;
};

#endif
