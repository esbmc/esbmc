/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_BASIC_SYMEX_H
#define CPROVER_BASIC_SYMEX_H

#include <map>
#include <set>

#include <options.h>
#include <namespace.h>
#include <replace_expr.h>
#include <std_code.h>

#include "execution_state.h"
#include "symex_target.h"
#include "goto_symex_state.h"

class basic_symext
{
public:
  basic_symext(
    const namespacet &_ns,
    contextt &_new_context,
    symex_targett &_target):
    constant_propagation(true),
    new_context(_new_context),
    ns(_ns),
    target(&_target)
  {
    options.set_option("simplify", true);
  }

  virtual ~basic_symext() { }

  typedef goto_symex_statet statet;

  virtual void symex(statet &state, execution_statet &ex_state, const codet &code, unsigned node_id);
  bool constant_propagation;

  optionst options;
  contextt &new_context;

protected:
  const namespacet &ns;
  symex_targett *target;

  virtual void do_simplify(exprt &expr);

  virtual void symex_block(statet &state, execution_statet &ex_state, const codet &code, unsigned node_id);
  virtual void symex_assign(statet &state, execution_statet &ex_state, const codet &code, unsigned node_id);

  void symex_assign_rec(statet &state, execution_statet &ex_state, const exprt &lhs, exprt &rhs, guardt &guard, unsigned node_id);
  void symex_assign_symbol(statet &state, execution_statet &ex_state, const exprt &lhs, exprt &rhs, guardt &guard,unsigned node_id);
  void symex_assign_typecast(statet &state, execution_statet &ex_state, const exprt &lhs, exprt &rhs, guardt &guard,unsigned node_id);
  void symex_assign_array(statet &state, execution_statet &ex_state, const exprt &lhs, exprt &rhs, guardt &guard,unsigned node_id);
  void symex_assign_member(statet &state, execution_statet &ex_state, const exprt &lhs, exprt &rhs, guardt &guard,unsigned node_id);
  void symex_assign_if(statet &state, execution_statet &ex_state, const exprt &lhs, exprt &rhs, guardt &guard,unsigned node_id);
  void symex_assign_byte_extract(statet &state,  execution_statet &ex_state, const exprt &lhs, exprt &rhs, guardt &guard,unsigned node_id);

  virtual void symex_malloc (statet &state, const exprt &lhs, const side_effect_exprt &code, execution_statet &ex_state, unsigned node_id);
  virtual void symex_cpp_delete    (statet &state, const codet &code);
  virtual void symex_cpp_new (statet &state, const exprt &lhs, const side_effect_exprt &code, execution_statet &ex_state, unsigned node_id);
  virtual void symex_fkt           (statet &state, const code_function_callt &code);
  virtual void symex_macro         (statet &state, const code_function_callt &code);
  virtual void symex_trace         (statet &state, const code_function_callt &code,unsigned node_id);
  virtual void symex_printf        (statet &state, const exprt &lhs, const exprt &code,unsigned node_id);

  static unsigned nondet_count;
  static unsigned dynamic_counter;

  void read(exprt &expr);
  virtual void replace_dynamic_allocation(const statet &state, exprt &expr);
  void replace_nondet(exprt &expr);

  void assignment(
    execution_statet &ex_state,
    const exprt &lhs,
    exprt &rhs);
};

void basic_symex(
  const codet &code,
  const namespacet &ns,
  symex_targett &target,
  execution_statet &ex_state,
  goto_symex_statet &state);

void basic_symex(
  const codet &code,
  const namespacet &ns,
  symex_targett &target,
  execution_statet &ex_state);

#endif
