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
  optionst &options,
  message_handlert& message_handler)
{
  get_global_vars(context);

  Forall_goto_functions(it, goto_functions)
    if(it->second.body_available)
      goto_k_inductiont(
        it->second,
        context,
        options,
        message_handler);

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
    if(!it->second.is_infinite_loop())
      continue;

    // We're going to change the code, so enable inductive step
    options.set_option("disable-inductive-step", false);

    // Start the loop conversion
    convert_loop(it->second);
  }
}

void goto_k_inductiont::convert_loop(loopst &loop)
{
  assert(!loop.get_goto_program().instructions.empty());

  // First, we need to fill the state member with the variables
  fill_state(loop);

  // Now, create the symbol and add them to the context
  // The states must be filled before the creation of the symbols
  // so the created symbol contain all variables in it.
  create_symbols();

  // Get current loop head and loop exit
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();

  // Create the nondet assignments on the beginning of the loop
  make_nondet_assign(loop_head);

  // Create the kindice variable and initialize it
  init_k_indice(loop_head);

  // Update the state vector, this will be inserted one instruction
  // after the loop head
  update_state_vector(loop_head);

  // Convert the loop body instructions
  convert_loop_body(loop);

  // Assign current state at the end of the loop
  assign_current_state(loop_exit);

  // Assume states
  assume_state_vector(loop_exit);

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

  if(constrain_all_states)
  {
    symbolt tmp_counter;
    tmp_counter.name="tmp_counter$"+i2string(i);
    tmp_counter.base_name="tmp_counter$"+i2string(i);
    tmp_counter.type=uint_type();
    tmp_counter.static_lifetime=true;
    tmp_counter.lvalue=true;

    context.move(tmp_counter, symbol_ptr);
  }

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

void goto_k_inductiont::init_k_indice(goto_programt::targett& loop_head)
{
  goto_programt dest;

  std::string identifier;
  identifier = "kindice$"+i2string(state_counter);
  exprt lhs_index = symbol_exprt(identifier, int_type());
  exprt zero_expr = gen_zero(int_type());
  code_assignt new_assign(lhs_index,zero_expr);
  copy(new_assign, ASSIGN, dest);

  goto_function.body.destructive_insert(loop_head, dest);
}

void goto_k_inductiont::update_state_vector(goto_programt::targett& loop_head)
{
  goto_programt dest;

  std::string identifier;
  identifier = "kindice$"+i2string(state_counter);

  array_typet state_vector;
  state_vector.subtype() = state;

  exprt lhs_index = symbol_exprt(identifier, int_type());
  exprt new_expr(exprt::with, state_vector);
  exprt lhs_array("symbol", state);
  exprt rhs("symbol", state);

  std::string identifier_lhs, identifier_rhs;
  identifier_lhs = "s$"+i2string(state_counter);
  identifier_rhs = "cs$"+i2string(state_counter);

  lhs_array.identifier(identifier_lhs);
  rhs.identifier(identifier_rhs);

  //s[k]=cs
  new_expr.reserve_operands(3);
  new_expr.copy_to_operands(lhs_array);
  new_expr.copy_to_operands(lhs_index);
  new_expr.move_to_operands(rhs);
  code_assignt new_assign(lhs_array,new_expr);
  copy(new_assign, ASSIGN, dest);

  // The update vector should be added one instruction after the loop head
  goto_programt::targett head_plus_one = loop_head;
  head_plus_one++;

  goto_function.body.destructive_insert(head_plus_one, dest);
}

void goto_k_inductiont::assign_current_state(goto_programt::targett& loop_exit)
{
  goto_programt dest;

  unsigned int component_size = state.components().size();
  for (unsigned int j = 0; j < component_size; j++)
  {
    exprt rhs_expr(state.components()[j]);
    exprt new_expr(exprt::with, state);
    exprt lhs_expr("symbol", state);

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

  goto_function.body.destructive_insert(loop_exit, dest);
}

void goto_k_inductiont::assume_all_state_vector(goto_programt::targett& loop_exit)
{
  goto_programt dest;

  // Temp symbol that will be used to count up to kindice
  std::string identifier;
  identifier = "tmp_counter$"+i2string(state_counter);
  exprt tmp_symbol = symbol_exprt(identifier, int_type());
  exprt zero_expr = gen_zero(int_type());
  code_assignt new_assign(tmp_symbol, zero_expr);
  copy(new_assign, ASSIGN, dest);

  // Condition (tmp_symbol <= kindice)
  exprt lhs_index =
      symbol_exprt("kindice$"+i2string(state_counter), int_type());
  exprt cond("<=", typet("bool"));
  cond.copy_to_operands(tmp_symbol, lhs_index);

  // do the v label
  goto_programt tmp_v;
  goto_programt::targett v=tmp_v.add_instruction();

  goto_programt tmp_z;
  goto_programt::targett z=tmp_z.add_instruction(ASSIGN);

  // kindice=kindice+1
  exprt one_expr = gen_one(int_type());
  exprt rhs_expr = gen_binary(exprt::plus, int_type(), lhs_index, one_expr);
  code_assignt kindice_plus(lhs_index,rhs_expr);
  expr2tc kindice_plus2;
  migrate_expr(kindice_plus, kindice_plus2);
  z->code = kindice_plus2;

  // do the x label
  rhs_expr = gen_binary(exprt::plus, int_type(), tmp_symbol, one_expr);
  code_assignt tmp_symbol_plus(tmp_symbol, rhs_expr);

  // do the u label
  goto_programt::targett u=v;

  // v: if(!c) goto z;
  v->make_goto(z);
  expr2tc tmp_cond;
  migrate_expr(cond, tmp_cond);
  tmp_cond = not2tc(tmp_cond);
  v->guard = tmp_cond;

  // do the w label
  goto_programt tmp_w;

  //set the type of the state vector
  array_typet state_vector;
  state_vector.subtype() = state;

  exprt new_expr(exprt::index, state);
  exprt lhs_array("symbol", state_vector);
  exprt rhs("symbol", state);

  lhs_array.identifier("s$"+i2string(state_counter));
  rhs.identifier("cs$"+i2string(state_counter));

  //s[k]
  new_expr.reserve_operands(2);
  new_expr.copy_to_operands(lhs_array);
  new_expr.copy_to_operands(tmp_symbol);

  //assume(s[k]!=cs)
  exprt result_expr = gen_binary(exprt::notequal, bool_typet(), new_expr, rhs);
  assume_cond(result_expr, false, tmp_w);

  // y: goto u;
  goto_programt tmp_y;
  goto_programt::targett y=tmp_y.add_instruction();
  y->make_goto(u);
  y->guard = true_expr;

  dest.destructive_append(tmp_v);
  dest.destructive_append(tmp_w);
  copy(tmp_symbol_plus, ASSIGN, dest);
  dest.destructive_append(tmp_y);
  dest.destructive_append(tmp_z);

  goto_function.body.destructive_insert(loop_exit, dest);
}

void goto_k_inductiont::assume_state_vector(
  goto_programt::targett& loop_exit)
{
  goto_programt dest;

  if(constrain_all_states)
  {
    assume_all_state_vector(loop_exit);
    return;
  }

  std::string identifier;
  identifier = "kindice$"+i2string(state_counter);

  array_typet state_vector;
  state_vector.subtype() = state;

  exprt lhs_index = symbol_exprt(identifier, int_type());
  exprt new_expr(exprt::index, state);
  exprt lhs_array("symbol", state_vector);
  exprt rhs("symbol", state);

  std::string identifier_lhs, identifier_rhs;

  identifier_lhs = "s$"+i2string(state_counter);
  identifier_rhs = "cs$"+i2string(state_counter);

  lhs_array.identifier(identifier_lhs);
  rhs.identifier(identifier_rhs);

  // s[k]
  new_expr.reserve_operands(2);
  new_expr.copy_to_operands(lhs_array);
  new_expr.copy_to_operands(lhs_index);

  // assume(s[k]!=cs)
  exprt result_expr = gen_binary(exprt::notequal, bool_typet(), new_expr, rhs);
  assume_cond(result_expr, false, dest);

  // TODO: This should be in a separate method
  // kindice=kindice+1
  exprt one_expr = gen_one(int_type());
  exprt rhs_expr = gen_binary(exprt::plus, int_type(), lhs_index, one_expr);
  code_assignt new_assign_plus(lhs_index, rhs_expr);
  copy(new_assign_plus, ASSIGN, dest);

  goto_function.body.destructive_insert(loop_exit, dest);
}

void goto_k_inductiont::convert_loop_body(loopst &loop)
{
  // Get loop head and loop exit
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();

  // Increment loop_head so we don't mistakenly convert the loop condition
  ++loop_head;

  // Iterate over the loop body and convert the guard of the goto instructions
  while(loop_head != loop_exit)
  {
    if(loop_head->is_goto())
    {
      exprt guard = migrate_expr_back(loop_head->guard);
      replace_guard(loop, guard);
      migrate_expr(guard, loop_head->guard);
    }

    ++loop_head;
  }
}

void goto_k_inductiont::replace_guard(loopst &loop, exprt& expr)
{
  Forall_operands(it, expr)
    replace_guard(loop, *it);

  if(loop.is_loop_var(expr))
    replace_by_cs_member(expr);
}

void goto_k_inductiont::replace_by_cs_member(exprt& expr)
{
  exprt lhs_struct("symbol", state);
  lhs_struct.identifier("cs$" + i2string(state_counter));

  exprt new_expr(exprt::member, expr.type());
  new_expr.reserve_operands(1);
  new_expr.copy_to_operands(lhs_struct);
  new_expr.identifier(expr.identifier());
  new_expr.component_name(expr.identifier());

  assert(!new_expr.get_string("component_name").empty());

  expr = new_expr;
}

void goto_k_inductiont::copy(const codet& code,
  goto_program_instruction_typet type,
  goto_programt& dest)
{
  goto_programt::targett t=dest.add_instruction(type);
  migrate_expr(code, t->code);
  t->location=code.location();
}

void goto_k_inductiont::assume_cond(
  const exprt& cond,
  const bool& neg,
  goto_programt& dest)
{
  goto_programt tmp_e;
  goto_programt::targett e=tmp_e.add_instruction(ASSUME);
  exprt result_expr = cond;
  if (neg)
    result_expr.make_not();

  migrate_expr(result_expr, e->guard);
  dest.destructive_append(tmp_e);
}
