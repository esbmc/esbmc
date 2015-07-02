/*
 * goto_unwind.cpp
 *
 *  Created on: Jun 3, 2015
 *      Author: mramalho
 */

#include <util/std_expr.h>
#include <util/expr_util.h>
#include <ansi-c/c_types.h>

#include <i2string.h>

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

  // Now, create the symbol and add them to the context
  // The states must be filled before the creation of the symbols
  // so the created symbol contain all variables in it.
  create_symbols();

  // Get current loop head
  goto_programt::targett loop_head;
  for(goto_programt::instructionst::iterator
      it=goto_function.body.instructions.begin();
      it!=goto_function.body.instructions.end();
      it++)
  {
    if(it->location_number ==
        loop.get_goto_program().instructions.begin()->location_number)
    {
      loop_head = it;
      break;
    }
  }

  // Create the nondet assignments on the beginning of the loop
  make_nondet_assign(loop_head);

  // We should clear the state by the end of the loop
  // This will be better encapsulated if we had an inductive step class
  // that inherit from loops where we could save all these information
  state.components().clear();

  // Finally, increment the state counter
  state_counter++;
}

void goto_k_inductiont::fill_state(loopst &loop)
{
  loopst::loop_varst loop_vars = loop.get_loop_vars();

  // State size will be the number of loop vars + global vars
  state.components().resize(loop_vars.size() + global_vars.size());

  // Copy from loop vars
  loopst::loop_varst::iterator it = loop_vars.begin();
  unsigned int i=0;
  for(i=0; i<loop_vars.size(); i++, it++)
  {
    state.components()[i] = (struct_typet::componentt &) it->second;
    state.components()[i].set_name(it->second.identifier());
    state.components()[i].pretty_name(it->second.identifier());
  }

  // Copy from global vars
  it = global_vars.begin();
  for( ; (i-loop_vars.size())<global_vars.size(); i++, it++)
  {
    state.components()[i] = (struct_typet::componentt &) it->second;
    state.components()[i].set_name(it->second.identifier());
    state.components()[i].pretty_name(it->second.identifier());
  }
}

void goto_k_inductiont::create_symbols()
{
  // Create symbol for the state$vector
  symbolt *symbol_ptr=NULL;
  unsigned int i = state_counter;

  symbolt state_symbol;
  state_symbol.name="c::state$vector"+i2string(i);
  state_symbol.base_name="state$vector"+i2string(i);
  state_symbol.is_type=true;
  state_symbol.type=state;
  state_symbol.mode="C";
  state_symbol.module="main";
  state_symbol.pretty_name="struct state$vector"+i2string(i);

  context.move(state_symbol, symbol_ptr);

  // Create new symbol for this state
  // First is kindice
  symbolt kindice_symbol;
  kindice_symbol.name="kindice$"+i2string(i);
  kindice_symbol.base_name="kindice$"+i2string(i);
  kindice_symbol.type=uint_type();
  kindice_symbol.static_lifetime=true;
  kindice_symbol.lvalue=true;

  context.move(kindice_symbol, symbol_ptr);

  // Then state_vector s
  // Its type is incomplete array
  typet incomplete_array_type("incomplete_array");
  incomplete_array_type.subtype()=state;

  symbolt state_vector_symbol;
  state_vector_symbol.name="s$"+i2string(i);
  state_vector_symbol.base_name="s$"+i2string(i);
  state_vector_symbol.type=incomplete_array_type;
  state_vector_symbol.static_lifetime=true;
  state_vector_symbol.lvalue=true;

  context.move(state_vector_symbol, symbol_ptr);

  // Finally, the current state cs
  symbolt current_state_symbol;
  current_state_symbol.name="cs$"+i2string(i);
  current_state_symbol.base_name="cs$"+i2string(i);
  current_state_symbol.type=state;
  current_state_symbol.static_lifetime=true;
  current_state_symbol.lvalue=true;

  context.move(current_state_symbol, symbol_ptr);
}

void goto_k_inductiont::make_nondet_assign(goto_programt::targett &loop_head)
{
  goto_programt dest;

  unsigned int component_size = state.components().size();
  for (unsigned int j = 0; j < component_size; j++)
  {
    exprt rhs_expr = side_effect_expr_nondett(
      state.components()[j].type());
    exprt new_expr(exprt::with, state);
    exprt lhs_expr("symbol", state);

    if (state.components()[j].type().is_array())
      rhs_expr = side_effect_expr_nondett(
        state.components()[j].type());

    std::string identifier;
    identifier = "cs$" + i2string(state_counter);
    lhs_expr.identifier(identifier);

    new_expr.reserve_operands(3);
    new_expr.copy_to_operands(lhs_expr);
    new_expr.copy_to_operands(exprt("member_name"));
    new_expr.move_to_operands(rhs_expr);

    if (!state.components()[j].has_operands())
    {
      new_expr.op1().component_name(state.components()[j].identifier());
      assert(!new_expr.op1().get_string("component_name").empty());
    }
    else
    {
      forall_operands(it, state.components()[j])
      {
        new_expr.op1().component_name(it->identifier());
        assert(!new_expr.op1().get_string("component_name").empty());
      }
    }

    code_assignt new_assign(lhs_expr, new_expr);
    copy(new_assign, ASSIGN, dest);
  }

  goto_function.body.destructive_insert(loop_head, dest);
}

void goto_k_inductiont::copy(const codet& code,
  goto_program_instruction_typet type,
  goto_programt& dest)
{
  goto_programt::targett t=dest.add_instruction(type);
  migrate_expr(code, t->code);
  t->location=code.location();
}
