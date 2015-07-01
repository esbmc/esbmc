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
  // Full unwind the program
  for(function_loopst::reverse_iterator
    it = function_loops.rbegin();
    it != function_loops.rend();
    ++it)
  {
    assert(!it->second.get_goto_program().empty());
    convert_loop(it->second);
  }
}

void goto_k_inductiont::convert_loop(loopst &loop)
{
  // TODO: check infinite/nondet loop
  assert(!loop.get_goto_program().instructions.empty());

  // First, we need to fill the state member with the variables
  fill_state(loop);

  // We should clear the state by the end of the loop
  // This will be better encapsulated if we had an inductive step class
  // that inherit from loops where we could save all these information
  state.components().clear();
}

void goto_k_inductiont::fill_state(loopst &loop)
{
  loopst::loop_varst loop_vars = loop.get_loop_vars();

  // State size will be the number of loop vars + global vars
  state.components().resize(loop_vars.size() + global_vars.size());

  // Copy from loop vars
  loopst::loop_varst::iterator it = loop_vars.begin();
  for(unsigned int i=0; i<loop_vars.size(); i++, it++)
  {
    state.components()[i] = (struct_typet::componentt &) it->second;
    state.components()[i].set_name(it->second.identifier());
    state.components()[i].pretty_name(it->second.identifier());
  }

  // Copy from global vars
  it = global_vars.begin();
  for(unsigned int i=0; i<global_vars.size(); i++, it++)
  {
    state.components()[i] = (struct_typet::componentt &) it->second;
    state.components()[i].set_name(it->second.identifier());
    state.components()[i].pretty_name(it->second.identifier());
  }
}
