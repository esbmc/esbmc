/*******************************************************************\

Module: Symbolic Execution - Stack tracking

Authors: Rafael Menezes and Lucas Cordeiro

Date: April 2020

\*******************************************************************/

#include <cassert>
#include <goto-symex/goto_symex.h>
#include <util/expr_util.h>
#include <util/irep2.h>

lessthanequal2tc goto_symex_statet::framet::process_stack_size(
  const expr2tc &expr,
  unsigned long stack_limit)
{
  // Store the total number of bits for a given stack frame.
  stack_frame_total += expr->type->get_width();

  // Create two constants to define stack frame size and stack limit.
  BigInt f_size(stack_frame_total);
  BigInt s_size(stack_limit);
  constant_int2tc function_irep2(type_pool.get_uint64(), f_size);
  constant_int2tc limit_irep2(type_pool.get_uint64(), s_size);

  // Ensure that the stack frame size is smaller than the stack limit.
  lessthanequal2tc check(function_irep2, limit_irep2);

  return check;
}

void goto_symex_statet::framet::decrease_stack_frame_size(const expr2tc &expr)
{
  const code_decl2t &decl_code = to_code_decl2t(expr);

  // Obtain the width of the dead expression and decrease it from the
  // total number of bits for a given stack frame.
  stack_frame_total -= decl_code.type->get_width();
}
