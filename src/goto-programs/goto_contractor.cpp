//
// Created by Mohannad Aldughaim on 09/01/2022.
//

#include "goto_contractor.h"

void goto_contractort::get_constraints(goto_functionst goto_functions)
{
  auto function = goto_functions.function_map.find("c:@F@main");
  auto it = function->second.body.instructions.begin();
  while(it != function->second.body.instructions.end())
  {
    if(it->is_assert())
      constraint = create_constraint_from_expr2t(it->guard);

    it++;
  }
}

void goto_contractort::get_intervals(goto_functionst goto_functions)
{
  //only main here
  auto function = goto_functions.function_map.find("c:@F@main");
  //std::cout << "\n parsing assumes\nfound main: " << function->first << "\n";
  auto it = function->second.body.instructions.begin();

  while(it != function->second.body.instructions.end())
  {
    if(it->is_assume())
      parse_intervals(it->guard);

    it++;
  }
}

void goto_contractort::parse_intervals(irep_container<expr2t> expr)
{
  symbol2tc symbol;
  double value;

  if(!is_comp_expr(expr))
    return;

  std::shared_ptr<relation_data> rel;
  rel = dynamic_pointer_cast<relation_data>(expr);

  //side1 1 is a symbol or typecast to symbol
  auto side1 = rel->side_1;
  //side 2 is always a number.
  auto side2 = rel->side_2;

  if(is_typecast2t(side1))
    if(is_symbol2t(to_typecast2t(side1).from))
      symbol = to_symbol2t(to_typecast2t(side1).from);
    else
      return;
  else if(is_symbol2t(side1))
    symbol = to_symbol2t(side1);
  else
    return;

  if(!is_constant_int2t(side2) && !is_constant_floatbv2t(side2))
    return;
  if(is_constant_int2t(side2))
    value = to_constant_int2t(side2).as_long();
  else
    return;

  int index = map->find(symbol->get_symbol_name());
  if(index == -1)
    return;
  switch(expr->expr_id)
  {
  case expr2t::expr_ids::greaterthan_id:
  case expr2t::expr_ids::greaterthanequal_id:
    map->update_lb_interval(value, index);
    break;
  case expr2t::expr_ids::lessthan_id:
  case expr2t::expr_ids::lessthanequal_id:
    map->update_ub_interval(value, index);
    break;
  case expr2t::expr_ids::equality_id:
    map->update_lb_interval(value, index);
    map->update_ub_interval(value, index);
    break;
  default:;
    //error
  }
}

void goto_contractort::insert_assume(
  goto_functionst goto_functions,
  IntervalVector new_intervals)
{
  message_handler.status("Inserting assumes.. ");

  loopst loop;
  for(auto &function_loop : function_loops)
    loop = function_loop;

  goto_programt::targett t = loop.get_original_loop_head();
  auto loop_exit = loop.get_original_loop_exit();

  goto_programt dest(message_handler);

  for(; t != loop_exit; t++)
    ;

  auto it = goto_functions.function_map.find("c:@F@main");
  auto goto_function = it->second;

  for(int i = 0; i < map->size(); i++)
  {
    symbol2tc X = map->symbols[i];
    if(
      isfinite(new_intervals[i].lb()) &&
      map->intervals[i].lb() != new_intervals[i].lb())
    {
      auto lb = create_signed_32_value_expr(new_intervals[i].lb());
      auto cond = create_greaterthanequal_relation(X, lb);
      goto_programt tmp_e(message_handler);
      goto_programt::targett e = tmp_e.add_instruction(ASSUME);
      e->inductive_step_instruction = false; //config.options.is_kind();
      e->guard = cond;
      e->location = loop_exit->location;
      goto_function.body.destructive_insert(loop_exit, tmp_e);
    }
    if(
      isfinite(new_intervals[i].ub()) &&
      map->intervals[i].ub() != new_intervals[i].ub())
    {
      auto ub = create_signed_32_value_expr(new_intervals[i].ub());
      auto cond = create_lessthanequal_relation(X, ub);
      goto_programt tmp_e(message_handler);
      goto_programt::targett e = tmp_e.add_instruction(ASSUME);
      e->inductive_step_instruction = false; //config.options.is_kind();
      e->guard = cond;
      e->location = loop_exit->location;
      goto_function.body.destructive_insert(loop_exit, tmp_e);
    }
  }
}

IntervalVector goto_contractort::contractor()
{
  //TODO: replace cout with message.status
  NumConstraint c = *constraint;
  //message_handler.status("My Constraint:" + c );
  //std::cout << "My Function:" << c.f << std::endl;

  auto complement = get_complement(c.op);

  NumConstraint c2(c.f, complement);
  //std::cout << "My complement Constraint:" << c2 << std::endl;
  //std::cout << "My Function:" << c2.f << std::endl;

  CtcFwdBwd c_out(c);
  CtcFwdBwd c_in(c2);

  domains = map->intervals;
  //std::cout << "My domains:" << domains << std::endl;
  auto X = domains;

  IntervalVector *s_in;
  c_in.contract(X);
  int num = domains.diff(X, s_in);
  //std::cout << "My domains after Inner contractor:" << X << std::endl;
  //for(int i = 0; i < num; i++)
  //message_handler.debug( "s_in[%d]: %f",i,  s_in[i]));

  return X;
}
ibex::CmpOp goto_contractort::get_complement(ibex::CmpOp op)
{
  switch(op)
  {
  case GEQ:
    return LT;
  case GT:
    return LEQ;
  case LEQ:
    return GT;
  case LT:
    return GEQ;
  default:
    message_handler.status("cant process equal");
    //abort();
    break;
  }
  return GEQ;
}
// && ||
Ctc *goto_contractort::create_contractors_from_expr2t(irep_container<expr2t>)
{
  return nullptr;
}
//>=,<
NumConstraint *
goto_contractort::create_constraint_from_expr2t(irep_container<expr2t> expr)
{
  //TODO: change to switch
  NumConstraint *c = nullptr;
  if(is_arith_expr(expr) || is_constant_number(expr) || is_symbol2t(expr))
  {
    message_handler.status("NOT A CONSTRAINT ");
  }
  else if(is_greaterthanequal2t(expr))
  {
    Function *f, *g;
    f = create_function_from_expr2t(to_greaterthanequal2t(expr).side_1);
    g = create_function_from_expr2t(to_greaterthanequal2t(expr).side_2);
    c = new NumConstraint(*vars, (*f)(*vars) >= (*g)(*vars));
    return c;
  }
  else if(is_greaterthan2t(expr))
  {
    Function *f, *g;
    f = create_function_from_expr2t(to_greaterthan2t(expr).side_1);
    g = create_function_from_expr2t(to_greaterthan2t(expr).side_2);
    c = new NumConstraint(*vars, (*f)(*vars) > (*g)(*vars));
    return c;
  }
  else if(is_lessthanequal2t(expr))
  {
    Function *f, *g;
    f = create_function_from_expr2t(to_lessthanequal2t(expr).side_1);
    g = create_function_from_expr2t(to_lessthanequal2t(expr).side_2);
    c = new NumConstraint(*vars, (*f)(*vars) <= (*g)(*vars));
    return c;
  }
  else if(is_lessthan2t(expr))
  {
    Function *f, *g;
    f = create_function_from_expr2t(to_lessthan2t(expr).side_1);
    g = create_function_from_expr2t(to_lessthan2t(expr).side_2);
    c = new NumConstraint(*vars, (*f)(*vars) < (*g)(*vars));
    return c;
  }
  else if(is_equality2t(expr))
  {
    Function *f, *g;
    f = create_function_from_expr2t(to_equality2t(expr).side_1);
    g = create_function_from_expr2t(to_equality2t(expr).side_2);
    c = new NumConstraint(*vars, (*f)(*vars) = (*g)(*vars));
    return c;
  }
  else if(is_notequal2t(expr))
  {
    Function *f, *g;
    f = create_function_from_expr2t(to_notequal2t(expr).side_1);
    g = create_function_from_expr2t(to_notequal2t(expr).side_2);
    c = new NumConstraint(*vars, (*f)(*vars) = (*g)(*vars));
    return c;
  }
  //>,<,<=
  return nullptr;
}
//+-*/
Function *
goto_contractort::create_function_from_expr2t(irep_container<expr2t> expr)
{
  Function *f = nullptr;
  Function *g, *h;
  
  expr->dump();
  //TODO: change to switch
  if(is_add2t(expr))
  {
    g = create_function_from_expr2t(to_add2t(expr).side_1);
    h = create_function_from_expr2t(to_add2t(expr).side_2);
    f = new Function(*vars, (*g)(*vars) + (*h)(*vars));
  }
  else if(is_sub2t(expr))
  {
    g = create_function_from_expr2t(to_add2t(expr).side_1);
    h = create_function_from_expr2t(to_add2t(expr).side_2);
    f = new Function(*vars, (*g)(*vars) - (*h)(*vars));
  }
  else if(is_mul2t(expr))
  {
    g = create_function_from_expr2t(to_mul2t(expr).side_1);
    h = create_function_from_expr2t(to_mul2t(expr).side_2);
    f = new Function(*vars, (*g)(*vars) * (*h)(*vars));
  }
  else if(is_div2t(expr))
  {
    g = create_function_from_expr2t(to_mul2t(expr).side_1);
    h = create_function_from_expr2t(to_mul2t(expr).side_2);
    f = new Function(*vars, (*g)(*vars) / (*h)(*vars));
  }
  else if(is_symbol2t(expr))
  {
    int index = create_variable_from_expr2t(expr);
    if(index != -1)
      f = new Function(*vars, (*vars)[index]);
    else
    {
      message_handler.error("ERROR: MAX VAR SIZE REACHED");
      exit(1);
    }
  }
  else if(is_typecast2t(expr))
  {
    f = create_function_from_expr2t(to_typecast2t(expr).from);
  }
  else if(is_constant_int2t(expr))
  {
    //f = new Function(*vars, to_constant_int2t(expr).value.to_int64());
  }
  else if(is_comp_expr(expr))
  {
    //Abort contractor
  }
  return f;
}

int goto_contractort::create_variable_from_expr2t(irep_container<expr2t> expr)
{
  std::string var_name = to_symbol2t(expr).get_symbol_name().c_str();
  int index = map->find(var_name);
  if(index == -1)
  {
    int new_index = map->add_var(var_name, to_symbol2t(expr));
    map->dump();
    if(new_index != -1)
      return new_index;
    return -1;
  }
  return index;
}

bool goto_contractort::run1()
{
  auto it = goto_functions.function_map.find("c:@F@main");
  {
    number_of_functions++;
    runOnFunction(*it);
    if(it->second.body_available)
    {
      const messaget msg;
      goto_loopst goto_loops(it->first, goto_functions, it->second, msg);
      auto function_loops = goto_loops.get_loops();
      this->function_loops = function_loops;
    }
  }
  goto_functions.update();
  return true;
}