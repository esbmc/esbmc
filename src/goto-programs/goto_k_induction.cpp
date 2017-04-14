/*
 * goto_unwind.cpp
 *
 *  Created on: Jun 3, 2015
 *      Author: mramalho
 */

#include <goto-programs/goto_k_induction.h>
#include <goto-programs/remove_skip.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/std_expr.h>

static unsigned int state_counter = 1;

void goto_k_induction(
  goto_functionst& goto_functions,
  contextt &context,
  optionst &options,
  message_handlert& message_handler)
{
  Forall_goto_functions(it, goto_functions)
    if(it->second.body_available)
      goto_k_inductiont(
        it->first,
        goto_functions,
        it->second,
        context,
        options,
        message_handler);

  goto_functions.update();
}

void goto_k_inductiont::goto_k_induction()
{
  // Full unwind the program
  for(function_loopst::iterator
    it = function_loops.begin();
    it != function_loops.end();
    ++it)
  {
    // TODO: Can we check if the loop is infinite? If so, we should
    // disable the forward condition

    // Start the loop conversion
    convert_finite_loop(*it);
  }
}

void goto_k_inductiont::convert_finite_loop(loopst& loop)
{
  // Get current loop head and loop exit
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();

  exprt loop_cond;
  get_loop_cond(loop_head, loop_exit, loop_cond);

  // If we didn't find a loop condition, don't change anything
  if(loop_cond.is_nil())
  {
    std::cout << "**** WARNING: we couldn't find a loop condition for the"
              << " following loop, so we're not converting it."
              << std::endl << "Loop: ";
    loop.dump();
    return;
  }

  // Fill the state member with the variables
  fill_state(loop);

  // Assume the loop condition before go into the loop
  assume_loop_cond_before_loop(loop_head, loop_cond);

  // Create the nondet assignments on the beginning of the loop
  make_nondet_assign(loop_head);

  // Get original head again
  // Since we are using insert_swap to keep the targets, the
  // original loop head as shifted to after the assume cond
  while((++loop_head)->inductive_step_instruction);

  // Check if the loop exit needs to be updated
  // We must point to the assume that was inserted in the previous
  // transformation
  adjust_loop_head_and_exit(loop_head, loop_exit);
}

void goto_k_inductiont::get_loop_cond(
  goto_programt::targett& loop_head,
  goto_programt::targett& loop_exit,
  exprt& loop_cond)
{
  loop_cond = nil_exprt();

  // Let's not change the loop head
  goto_programt::targett tmp = loop_head;

  // Look for an loop condition
  while(!tmp->is_goto())
    ++tmp;

  // If we hit the loop's end and didn't find any loop condition
  // return a nil exprt
  if(tmp == loop_exit)
    return;

  // Otherwise, fill the loop condition
  loop_cond = migrate_expr_back(tmp->guard);
}

void goto_k_inductiont::make_nondet_assign(goto_programt::targett& loop_head)
{
  goto_programt dest;

  unsigned int component_size = state.components().size();
  for (unsigned int j = 0; j < component_size; j++)
  {
    exprt rhs_expr = side_effect_expr_nondett(
      state.components()[j].type());
    exprt lhs_expr = state.components().at(j);

    code_assignt new_assign(lhs_expr, rhs_expr);
    new_assign.location() = loop_head->location;
    copy(new_assign, ASSIGN, dest);
  }

  goto_function.body.insert_swap(loop_head, dest);
}

void goto_k_inductiont::assume_loop_cond_before_loop(
  goto_programt::targett& loop_head,
  exprt &loop_cond)
{
  goto_programt dest;

  if(loop_cond.is_not())
    assume_cond(loop_cond.op0(), dest);
  else
    assume_cond(loop_cond, dest);

  goto_function.body.insert_swap(loop_head, dest);
}

void goto_k_inductiont::assume_neg_loop_cond_after_loop(
  goto_programt::targett& loop_exit,
  exprt& loop_cond)
{
  goto_programt dest;

  if(loop_cond.is_not())
    assume_cond(gen_not(loop_cond.op0()), dest);
  else
    assume_cond(gen_not(loop_cond), dest);

  goto_programt::targett _loop_exit = loop_exit;
  ++_loop_exit;

  goto_function.body.insert_swap(_loop_exit, dest);
}

void goto_k_inductiont::adjust_loop_head_and_exit(
  goto_programt::targett& loop_head,
  goto_programt::targett& loop_exit)
{
  loop_exit->targets.clear();
  loop_exit->targets.push_front(loop_head);

  goto_programt::targett _loop_exit = loop_exit;
  ++_loop_exit;

  // Zero means that the instruction was added during
  // the k-induction transformation
  if(_loop_exit->location_number == 0)
  {
    // Clear the target
    loop_head->targets.clear();

    // And set the target to be the newly inserted assume(cond)
    loop_head->targets.push_front(_loop_exit);
  }
}

// Duplicate the loop after loop_exit, but without the backward goto
void goto_k_inductiont::duplicate_loop_body(
  goto_programt::targett& loop_head,
  goto_programt::targett& loop_exit)
{
  goto_programt::targett _loop_exit = loop_exit;
  ++_loop_exit;

  // Iteration points will only be duplicated
  std::vector<goto_programt::targett> iteration_points;
  iteration_points.resize(2);

  if(_loop_exit != loop_head)
  {
    goto_programt::targett t_before = _loop_exit;
    t_before--;

    if(t_before->is_goto() && is_true(t_before->guard))
    {
      // no 'fall-out'
    }
    else
    {
      // guard against 'fall-out'
      goto_programt::targett t_goto = goto_function.body.insert(_loop_exit);

      t_goto->make_goto(_loop_exit);
      t_goto->location = _loop_exit->location;
      t_goto->function = _loop_exit->function;
      t_goto->guard = gen_true_expr();
    }
  }

  goto_programt::targett t_skip = goto_function.body.insert(_loop_exit);
  goto_programt::targett loop_iter = t_skip;

  t_skip->make_skip();
  t_skip->location = loop_head->location;
  t_skip->function = loop_head->function;

  // record the exit point of first iteration
  iteration_points[0] = loop_iter;

  // build a map for branch targets inside the loop
  std::map<goto_programt::targett, unsigned> target_map;

  {
    unsigned count = 0;
    for(goto_programt::targett t = loop_head; t != loop_exit; t++)
    {
      assert(t != goto_function.body.instructions.end());

      // Don't copy instructions inserted by the inductive-step
      // transformations
      if(t->inductive_step_instruction)
        continue;

      target_map[t] = count++;
    }
  }

  // we make k-1 copies, to be inserted before _loop_exit
  goto_programt copies;

  // make a copy
  std::vector<goto_programt::targett> target_vector;
  target_vector.reserve(target_map.size());

  for(goto_programt::targett t = loop_head; t != loop_exit; t++)
  {
    assert(t != goto_function.body.instructions.end());

    // Don't copy instructions inserted by the inductive-step
    // transformations
    if(t->inductive_step_instruction)
      continue;

    goto_programt::targett copied_t = copies.add_instruction();
    *copied_t = *t;
    target_vector.push_back(copied_t);
  }

  // record exit point of this copy
  iteration_points[1] = target_vector.back();

  // adjust the intra-loop branches
  for(unsigned i=0; i < target_vector.size(); i++)
  {
    goto_programt::targett t = target_vector[i];

    for(goto_programt::instructiont::targetst::iterator
        t_it = t->targets.begin();
        t_it != t->targets.end();
        t_it++)
    {
      std::map<goto_programt::targett, unsigned>::const_iterator m_it =
        target_map.find(*t_it);

      if(m_it != target_map.end()) // intra-loop?
      {
        assert(m_it->second < target_vector.size());
        *t_it = target_vector[m_it->second];
      }
    }
  }

  // now insert copies before _loop_exit
  goto_function.body.insert_swap(loop_exit, copies);

  // remove skips
  remove_skip(goto_function.body);
}

// Convert assert into assumes on the original loop (don't touch the
// copy made on the last step)
void goto_k_inductiont::convert_assert_to_assume(
  goto_programt::targett& loop_head,
  goto_programt::targett& loop_exit)
{
  for(goto_programt::targett t=loop_head; t!=loop_exit; t++)
    if(t->is_assert()) t->type=ASSUME;
}

void goto_k_inductiont::convert_infinite_loop(loopst &loop)
{
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
  make_nondet_state_assign(loop_head);

  // Create the kindice variable and initialize it
  init_k_indice(loop_head);

  // Update the state vector, this will insert one instruction
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
  state.components().resize(loop_vars.size());

  // Copy from loop vars
  loopst::loop_varst::iterator it = loop_vars.begin();
  unsigned int i=0;
  for(i=0; i<loop_vars.size(); i++, it++)
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

void goto_k_inductiont::make_nondet_state_assign(
  goto_programt::targett &loop_head)
{
  goto_programt dest;

  unsigned int component_size = state.components().size();
  for (unsigned int j = 0; j < component_size; j++)
  {
    exprt rhs_expr = side_effect_expr_nondett(
      state.components()[j].type());

    if (state.components()[j].type().is_array())
      rhs_expr = side_effect_expr_nondett(
        state.components()[j].type());

    with_exprt new_expr(gen_current_state(), exprt("member_name"), rhs_expr);

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

    code_assignt new_assign(gen_current_state(), new_expr);
    copy(new_assign, ASSIGN, dest);
  }

  goto_function.body.destructive_insert(loop_head, dest);
}

void goto_k_inductiont::init_k_indice(goto_programt::targett& loop_head)
{
  goto_programt dest;

  code_assignt new_assign(gen_kindice(), gen_zero(int_type()));
  copy(new_assign, ASSIGN, dest);

  goto_function.body.destructive_insert(loop_head, dest);
}

void goto_k_inductiont::update_state_vector(goto_programt::targett& loop_head)
{
  goto_programt dest;

  with_exprt new_expr(gen_state_vector(), gen_kindice(), gen_current_state());
  code_assignt new_assign(gen_state_vector(), new_expr);
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

    with_exprt new_expr(
      gen_current_state(), exprt("member_name"), state.components()[j]);

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

    code_assignt new_assign(gen_current_state(), new_expr);
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
  assume_cond(result_expr, tmp_w);

  // y: goto u;
  goto_programt tmp_y;
  goto_programt::targett y=tmp_y.add_instruction();
  y->make_goto(u);
  y->guard = gen_true_expr();

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
  if(constrain_all_states)
  {
    assume_all_state_vector(loop_exit);
    return;
  }

  goto_programt dest;

  // assume(s[k]!=cs)
  exprt result_expr =
    gen_binary(
      exprt::notequal,
      bool_typet(),
      gen_state_vector_indexed(gen_kindice()),
      gen_current_state());
  assume_cond(result_expr, dest);

  kindice_incr(dest);

  goto_function.body.destructive_insert(loop_exit, dest);
}

void goto_k_inductiont::convert_loop_body(loopst &loop)
{
  // Get loop head and loop exit
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();

  // Increment loop_head so we don't mistakenly convert the loop condition
  ++loop_head;

  // Iterate over the loop body and convert instructions
  while(loop_head != loop_exit)
  {
    convert_instruction(loop, loop_head, function_name);
    ++loop_head;
  }
}

void goto_k_inductiont::convert_instruction(
  loopst &loop,
  goto_programt::targett instruction,
  const irep_idt &_function_name)
{
  // Convert guards on the loop (if statements)
  if(instruction->is_goto())
  {
    exprt guard = migrate_expr_back(instruction->guard);
    replace_guard(loop, guard);
    migrate_expr(guard, instruction->guard);
  }
  // Look for ternary operator to be converted as well
  else if(instruction->is_assign())
  {
    exprt assignment = migrate_expr_back(instruction->code);
    assert(assignment.operands().size() == 2);
    replace_ternary(loop, assignment.op1());
    migrate_expr(assignment, instruction->code);
  }
  // If it is a function call, we have to iterate over its body
  else if(instruction->is_function_call())
  {
    code_function_call2t &function_call =
      to_code_function_call2t(instruction->code);

    // Don't do function pointers
    if(is_dereference2t(function_call.function))
      return;

    irep_idt &identifier = to_symbol2t(function_call.function).thename;

    // This means recursion, do nothing
    if(identifier == _function_name)
      return;

    // find code in function map
    goto_functionst::function_mapt::iterator it =
      goto_functions.function_map.find(identifier);

    if (it == goto_functions.function_map.end()) {
      std::cerr << "failed to find `" + id2string(identifier) +
          "' in function_map";
      abort();
    }

    // Avoid iterating over functions that don't have a body
    if(!it->second.body_available)
      return;

    for(goto_programt::instructionst::iterator head=
        it->second.body.instructions.begin();
        head != it->second.body.instructions.end();
        ++head)
    {
      convert_instruction(loop, head, identifier);
    }
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
  exprt new_expr(exprt::member, expr.type());
  new_expr.reserve_operands(1);
  new_expr.copy_to_operands(gen_current_state());
  new_expr.identifier(expr.identifier());
  new_expr.component_name(expr.identifier());

  assert(!new_expr.get_string("component_name").empty());

  expr = new_expr;
}

void goto_k_inductiont::replace_ternary(
  loopst& loop,
  exprt& expr,
  bool is_if_cond)
{
  Forall_operands(it, expr)
    replace_ternary(loop, *it, is_if_cond);

  if(expr.id()=="if")
    replace_ternary(loop, expr.op0(), true);

  if(loop.is_loop_var(expr) && is_if_cond)
    replace_by_cs_member(expr);
}

void goto_k_inductiont::copy(const codet& code,
  goto_program_instruction_typet type,
  goto_programt& dest)
{
  goto_programt::targett t=dest.add_instruction(type);
  t->inductive_step_instruction = true;

  migrate_expr(code, t->code);
  t->location=code.location();
}

void goto_k_inductiont::assume_cond(
  const exprt& cond,
  goto_programt& dest)
{
  goto_programt tmp_e;
  goto_programt::targett e=tmp_e.add_instruction(ASSUME);
  e->inductive_step_instruction = true;

  migrate_expr(cond, e->guard);
  dest.destructive_append(tmp_e);
}

void goto_k_inductiont::kindice_incr(goto_programt& dest)
{
  goto_programt tmp_z;
  goto_programt::targett z=tmp_z.add_instruction(ASSIGN);

  // kindice = kindice + 1
  exprt rhs_expr =
    gen_binary(exprt::plus, int_type(), gen_kindice(), gen_one(int_type()));
  code_assignt kindice_plus(gen_kindice(), rhs_expr);

  expr2tc kindice_plus2;
  migrate_expr(kindice_plus, kindice_plus2);

  z->code = kindice_plus2;
  dest.destructive_append(tmp_z);
}

symbol_exprt goto_k_inductiont::gen_kindice()
{
  return symbol_exprt("kindice$"+i2string(state_counter), int_type());
}

symbol_exprt goto_k_inductiont::gen_state_vector()
{
  return symbol_exprt("s$"+i2string(state_counter), array_typet(state));
}

symbol_exprt goto_k_inductiont::gen_current_state()
{
  return symbol_exprt("cs$"+i2string(state_counter), state);
}

exprt goto_k_inductiont::gen_state_vector_indexed(exprt index)
{
  exprt state_vector_indexed(exprt::index, state);

  state_vector_indexed.reserve_operands(2);
  state_vector_indexed.copy_to_operands(gen_state_vector());
  state_vector_indexed.copy_to_operands(index);

  return state_vector_indexed;
}
