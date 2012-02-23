/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_SYMEX_GOTO_SYMEX_H
#define CPROVER_GOTO_SYMEX_GOTO_SYMEX_H

#include <map>
#include <std_types.h>
#include <i2string.h>
#include <hash_cont.h>
#include <options.h>

#include <goto-programs/goto_functions.h>

#include "goto_symex_state.h"
#include "symex_target.h"

class reachability_treet; // Forward dec
class execution_statet; // Forward dec

class goto_symext
{
public:
  goto_symext(const namespacet &_ns, contextt &_new_context,
              symex_targett *_target, const optionst &opts);
  goto_symext(const goto_symext &sym);
  goto_symext& operator=(const goto_symext &sym);

  // Types

public:
  friend class symex_dereference_statet;
  friend class bmct;

  typedef goto_symex_statet statet;

  class symex_resultt {
  public:
    symex_resultt(symex_targett *t, unsigned int claims, unsigned int remain) :
      target(t), total_claims(claims), remaining_claims(remain) { };

    symex_targett *target;
    unsigned int total_claims;
    unsigned int remaining_claims;
  };

  // Macros
  //
  irep_idt guard_identifier(statet &state)
  {
	  return irep_idt(id2string(guard_identifier_s) + "!" + i2string(state.top().level1._thread_id));
  };

  // Methods

  symex_resultt *get_symex_result(void);

  void operator()(const goto_functionst &goto_functions);

  virtual void symex_step(
  const goto_functionst &goto_functions,
  reachability_treet & art);

protected:
  virtual void do_simplify(exprt &expr);

  void dereference(
    exprt &expr,
    statet &state,
    const bool write);

  void dereference_rec(
    exprt &expr,
    guardt &guard,
    class dereferencet &dereference,
    const bool write);

  // guards

  //irep_idt guard_identifier;
  irep_idt guard_identifier_s;

  // symex

  virtual void symex_goto(statet &state, const exprt &old_guard);

  void symex_return(statet &state, execution_statet &ex_state);

  void symex_other(
    const goto_functionst &goto_functions,
    statet &state,
    execution_statet &ex_state);

  virtual void claim(
    const exprt &expr,
    const std::string &msg,
    statet &state);

  virtual void assume(const exprt &assumption, statet &state);

  // gotos
  void merge_gotos(statet &state);

  void merge_value_sets(
    const statet::goto_statet &goto_state,
    statet &dest);

  void phi_function(const statet::goto_statet &goto_state, statet &state);

  bool get_unwind(
    const symex_targett::sourcet &source,
    unsigned unwind);

  void loop_bound_exceeded(statet &state, const exprt &guard);

  // function calls

  void pop_frame(statet &state);
  bool make_return_assignment(statet &state, execution_statet &ex_state,
                              code_assignt &assign, const code_returnt &code);

  void symex_function_call(
    const goto_functionst &goto_functions,
    execution_statet &state,
    const code_function_callt &call);

  void symex_end_of_function(statet &state);

  void symex_function_call_symbol(
    const goto_functionst &goto_functions,
    execution_statet &state,
    const code_function_callt &call);

  void symex_function_call_code(
    const goto_functionst &goto_functions,
    execution_statet &state,
    const code_function_callt &call);

  bool get_unwind_recursion(
    const irep_idt &identifier,
    unsigned unwind);

  void argument_assignments(
    const code_typet &function_type,
    statet &state,
    const exprt::operandst &arguments);

  void locality(
    unsigned frame_counter,
    statet &state,
    const goto_functionst::goto_functiont &goto_function);

  void add_end_of_function(
    exprt &code,
    const irep_idt &identifier);

  // dynamic stuff
  void replace_dynamic_allocation(const statet &state, exprt &expr);
  bool is_valid_object(const statet &state, const symbolt &symbol);

  virtual void symex_assign(statet &state, const codet &code);
  void symex_assign_rec(statet &state, const exprt &lhs, exprt &rhs, guardt &guard);
  void symex_assign_symbol(statet &state, const exprt &lhs, exprt &rhs, guardt &guard);
  void symex_assign_typecast(statet &state, const exprt &lhs, exprt &rhs, guardt &guard);
  void symex_assign_array(statet &state, const exprt &lhs, exprt &rhs, guardt &guard);
  void symex_assign_member(statet &state, const exprt &lhs, exprt &rhs, guardt &guard);
  void symex_assign_if(statet &state, const exprt &lhs, exprt &rhs, guardt &guard);
  void symex_assign_byte_extract(statet &state, const exprt &lhs, exprt &rhs, guardt &guard);

  void symex_malloc(statet &state, const exprt &lhs, const side_effect_exprt &code);
  void symex_cpp_delete(statet &state, const codet &code);
  void symex_cpp_new(statet &state, const exprt &lhs, const side_effect_exprt &code);
  void symex_macro(statet &state, const code_function_callt &code);
  void symex_printf(statet &state, const exprt &lhs, const exprt &code);

  void replace_nondet(exprt &expr);

  virtual unsigned int &get_dynamic_counter(void) = 0;
  virtual unsigned int &get_nondet_counter(void) = 0;

  // Members

  unsigned total_claims, remaining_claims;
  reachability_treet *art1;
  hash_set_cont<irep_idt, irep_id_hash> body_warnings;
  std::map<unsigned, long> unwind_set;
  unsigned int max_unwind;
  bool constant_propagation;
  const namespacet &ns;
  const optionst &options;
  contextt &new_context;
  symex_targett *target;
};

#endif
