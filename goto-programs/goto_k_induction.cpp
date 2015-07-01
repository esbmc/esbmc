/*
 * goto_unwind.cpp
 *
 *  Created on: Jun 3, 2015
 *      Author: mramalho
 */

#include <util/std_expr.h>
#include <util/expr_util.h>

#include "goto_k_induction.h"
#include "remove_skip.h"

loopst::loop_varst global_vars;
void add_global_vars(const exprt& expr)
{
  if (expr.is_symbol() && expr.type().id() != "code")
  {
    if(check_var_name(expr))
      global_vars.insert(
        std::pair<irep_idt, const exprt>(expr.identifier(), expr));
  }
  else
  {
    forall_operands(it, expr)
      add_global_vars(*it);
  }
}

void get_global_vars(contextt &context)
{
  forall_symbols(it, context.symbols) {
    if(it->second.static_lifetime && !it->second.type.is_pointer())
    {
      exprt s = symbol_expr(it->second);
      if(it->second.value.id()==irep_idt("array_of"))
        s.type()=it->second.value.type();
      add_global_vars(s);
    }
  }
}

void dump_global_vars()
{
  std::cout << "Loop variables:" << std::endl;

  u_int i = 0;
  for (std::pair<irep_idt, const exprt> expr : global_vars)
    std::cout << ++i << ". \t" << "identifier: " << expr.first << std::endl
    << " " << expr.second << std::endl << std::endl;
  std::cout << std::endl;
}

void goto_k_induction(
  goto_functionst& goto_functions,
  contextt &context,
  message_handlert& message_handler)
{
  get_global_vars(context);

  Forall_goto_functions(it, goto_functions)
    if(it->second.body_available)
      goto_k_inductiont(it->second, context, message_handler);

  goto_functions.update();
}

void goto_k_inductiont::goto_k_induction()
{
}
