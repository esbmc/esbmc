/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_SYMEX_GOTO_SYMEX_H
#define CPROVER_GOTO_SYMEX_GOTO_SYMEX_H

#include <std_types.h>
#include <goto-programs/goto_functions.h>

#include <i2string.h>
#include "basic_symex.h"
#include "reachability_tree.h"

class goto_symext:
  public basic_symext
{
public:
  goto_symext(
    const namespacet &_ns,
    contextt &_new_context,
    symex_targett &_target):
    basic_symext(_ns, _new_context, _target),
    total_claims(0),
    remaining_claims(0),
    guard_identifier_s("goto_symex::\\guard")
  {
    options.set_option("no-assertions", false);
    art1 = NULL;
  }

  ~goto_symext() {
    if (art1 != NULL)
      delete art1;
  }

  // all at once
  void operator()(
    const goto_functionst &goto_functions);

    bool restore_from_dfs_state(const reachability_treet::dfs_position &dfs);
    symex_target_equationt *multi_formulas_get_next_formula();
    bool multi_formulas_setup_next();
    void multi_formulas_init(const goto_functionst &goto_functions);
    void save_checkpoint(const std::string fname) const;

  void symex_step(
  const goto_functionst &goto_functions,
  reachability_treet & art);

protected:
  friend class symex_dereference_statet;
  reachability_treet *art1;

  // statistics
  unsigned total_claims, remaining_claims;

  void dereference(
    exprt &expr,
    statet &state,
    const bool write, unsigned node_id);

  void dereference_rec(
    exprt &expr,
    guardt &guard,
    class dereferencet &dereference,
    const bool write);

  // guards

  //irep_idt guard_identifier;
  irep_idt guard_identifier_s;

  irep_idt guard_identifier(statet &state)
  {
	  return irep_idt(id2string(guard_identifier_s) + "!" + i2string(state.top().level1._thread_id));
  };

  // symex

  void symex_goto(statet &state, execution_statet &ex_state, unsigned node_id);

  void symex_return(statet &state, execution_statet &ex_state, unsigned node_id);

  void symex_other(
    const goto_functionst &goto_functions,
    statet &state,
    execution_statet &ex_state,
    unsigned node_id);

  void claim(
    const exprt &expr,
    const std::string &msg,
    statet &state, unsigned node_id);

  // gotos
  void merge_gotos(statet &state, execution_statet &ex_state, unsigned node_id);

  void merge_value_sets(
    const statet::goto_statet &goto_state,
    statet &dest);

  void phi_function(
    const statet::goto_statet &goto_state,
    statet &state, execution_statet &ex_state, unsigned node_id);

  bool get_unwind(
    const symex_targett::sourcet &source,
    unsigned unwind);

  void loop_bound_exceeded(statet &state, const exprt &guard,unsigned node_id);

  // function calls

  void pop_frame(statet &state);
  void return_assignment(statet &state, execution_statet &ex_state, unsigned node_id);

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
    execution_statet &state,
    const exprt::operandst &arguments);

  void locality(
    unsigned frame_counter,
    statet &state,
    const goto_functionst::goto_functiont &goto_function,
    unsigned exec_node_id);

  void add_end_of_function(
    exprt &code,
    const irep_idt &identifier);

  // dynamic stuff
  void replace_dynamic_allocation(const statet &state, exprt &expr);
  bool is_valid_object(const statet &state, const symbolt &symbol);

  // Assignment methods
  void assignment(execution_statet &ex_state, const exprt &lhs, exprt &rhs);

  void symex_assign(statet &state, execution_statet &ex_state, const codet &code, unsigned node_id);
  void symex_assign_rec(statet &state, execution_statet &ex_state, const exprt &lhs, exprt &rhs, guardt &guard, unsigned node_id);
  void symex_assign_symbol(statet &state, execution_statet &ex_state, const exprt &lhs, exprt &rhs, guardt &guard,unsigned node_id);
  void symex_assign_typecast(statet &state, execution_statet &ex_state, const exprt &lhs, exprt &rhs, guardt &guard,unsigned node_id);
  void symex_assign_array(statet &state, execution_statet &ex_state, const exprt &lhs, exprt &rhs, guardt &guard,unsigned node_id);
  void symex_assign_member(statet &state, execution_statet &ex_state, const exprt &lhs, exprt &rhs, guardt &guard,unsigned node_id);
  void symex_assign_if(statet &state, execution_statet &ex_state, const exprt &lhs, exprt &rhs, guardt &guard,unsigned node_id);
  void symex_assign_byte_extract(statet &state,  execution_statet &ex_state, const exprt &lhs, exprt &rhs, guardt &guard,unsigned node_id);

  void symex_malloc(statet &state, const exprt &lhs, const side_effect_exprt &code, execution_statet &ex_state, unsigned node_id);
  void symex_cpp_delete(statet &state, const codet &code);
  void symex_cpp_new(statet &state, const exprt &lhs, const side_effect_exprt &code, execution_statet &ex_state, unsigned node_id);
  void symex_macro(statet &state, const code_function_callt &code);
  void symex_printf(statet &state, const exprt &lhs, const exprt &code,unsigned node_id);

  void replace_nondet(exprt &expr, execution_statet &ex_state);
};

#endif
