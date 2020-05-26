/*******************************************************************\

Module: Symbolic Execution - Stack tracking

Author: Rafael Sá Menezes

Date: April 2020

\*******************************************************************/

#include <cassert>
#include <goto-symex/goto_symex.h>
#include <util/expr_util.h>
#include <util/irep2.h>

void goto_symext::process_stack_size(expr2tc &expr)
{
  code_decl2t &decl_code = to_code_decl2t(expr);
  cur_state->top().frame_info.add_decl(decl_code);
  if(stack_limit != -1)
  {
    static unsignedbv_type2tc u32type(32);
    BigInt f_size(cur_state->top().frame_info.total);
    static BigInt s_size(stack_limit);
    constant_int2tc function_irep2(u32type, f_size);
    constant_int2tc limit_irep2(u32type, s_size);

    lessthanequal2tc check(function_irep2, limit_irep2);
    claim(check, "Stack limit property was violated");
    ;
  }
}