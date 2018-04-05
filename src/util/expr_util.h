/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/expr.h>
#include <util/std_types.h>
#include <util/symbol.h>

//
// WARNING: The following functions are depreciated, and will
//          eventually be removed
//
// Look at std_expr.h instead.
//

exprt gen_zero(const typet &type, bool array_as_array_of);
exprt gen_zero(const typet &type);
exprt gen_one(const typet &type);
exprt gen_not(const exprt &op);
exprt gen_unary(const std::string &id, const typet &type, const exprt &op);
exprt gen_binary(
  const std::string &id,
  const typet &type,
  const exprt &op1,
  const exprt &op2);
exprt gen_binary(
  irep_idt &id,
  const typet &type,
  const exprt &op1,
  const exprt &op2);
exprt gen_and(const exprt &op1, const exprt &op2);
exprt gen_and(const exprt &op1, const exprt &op2, const exprt &op3);
exprt gen_or(const exprt &op1, const exprt &op2);
exprt gen_or(const exprt &op1, const exprt &op2, const exprt &op3);
exprt gen_implies(const exprt &op1, const exprt &op2);
exprt gen_address_of(const exprt &op);

pointer_typet gen_pointer_type(const typet &subtype);

void gen_and(exprt &expr);
void gen_or(exprt &expr);

exprt symbol_expr(const symbolt &symbol);

void make_next_state(exprt &expr);

exprt make_binary(const exprt &src);
