//
// Created by Mohannad Aldughaim on 09/01/2022.
//

#include "goto_contractor.h"

using namespace std;
void goto_contractor(
  goto_functionst &goto_functions,
  const messaget &message_handler)
{
  Forall_goto_functions(it, goto_functions)
  {
    if(it->second.body_available)
    {
      std::cout << it->first << "\n";
      goto_contractort(it->first, goto_functions, it->second, message_handler);
    }
  }
  goto_functions.update();
}

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
  std::cout << "\n parsing assumes\nfound main: " << function->first << "\n";
  auto it = function->second.body.instructions.begin();

  while(it != function->second.body.instructions.end())
  {
    if(it->is_assume())
      parse_intervals(it->guard);

    it++;
  }
}

void goto_contractort::insert_assume(
  goto_functionst goto_functions,
  IntervalVector new_intervals)
{
  std::cout << "Inserting assumes.. " << std::endl;
  loopst loop;
  for(auto &function_loop : function_loops)
    loop = function_loop;

  goto_programt::targett t = loop.get_original_loop_head();
  auto loop_exit = loop.get_original_loop_exit();

  goto_programt dest(message_handler);

  for(; t != loop_exit; t++)
    ;

  for(int i = 0; i < map->size(); i++)
  {
    symbol2tc X = map->symbols[i];
    if(isfinite(new_intervals[i].lb()) && map->intervals[i].lb() != new_intervals[i].lb())
    {
      if(is_signedbv_type(X)){}
      auto lb = create_signed_32_value_expr(new_intervals[i].lb());
      auto cond = create_greaterthanequal_relation(X, lb);
      goto_programt tmp_e(message_handler);
      goto_programt::targett e = tmp_e.add_instruction(ASSUME);
      e->inductive_step_instruction = false; //config.options.is_kind();
      e->guard = cond;
      e->location = loop_exit->location;
      goto_function.body.destructive_insert(loop_exit, tmp_e);
    }
    if(isfinite(new_intervals[i].ub())&& map->intervals[i].ub() != new_intervals[i].ub())
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
  NumConstraint c = *constraint;
  std::cout << "My Constraint:" << c << std::endl;
  std::cout << "My Function:" << c.f << std::endl;

  auto complement = get_complement(c.op);

  NumConstraint c2(c.f, complement);
  std::cout << "My complement Constraint:" << c2 << std::endl;
  std::cout << "My Function:" << c2.f << std::endl;

  CtcFwdBwd c_out(c);
  CtcFwdBwd c_in(c2);

  domains = map->intervals;
  std::cout << "My domains:" << domains << std::endl;
  auto X = domains;

  IntervalVector *s_in;
  c_in.contract(X);
  int num = domains.diff(X, s_in);
  std::cout << "My domains after Inner contractor:" << X << std::endl;
  for(int i = 0; i < num; i++)
    std::cout << "s_in[" << i << "]:" << s_in[i] << std::endl;

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
    cout<<"cant process equal"<<endl;
    abort();
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
  NumConstraint *c = nullptr;
  if(is_arith_expr(expr) || is_constant_number(expr) || is_symbol2t(expr))
  {
    //Function *f=create_function_from_expr2t(expr);
    //NumConstraint *c= new NumConstraint(*vars,(*f)(*vars));
    cout << " NOT A CONSTRAINT " << endl;
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
  //auto op = get_expr_id(expr);
  Function *f = nullptr;
  Function *g, *h;
  cout << "dumping expression:" << endl;
  expr->dump();
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
  }
  else if(is_symbol2t(expr))
  {
    int index = create_variable_from_expr2t(expr);
    if(index != -1)
      f = new Function(*vars, (*vars)[index]);
    else
    {
      cout << "ERROR: MAX VAR SIZE REACHED" << endl;
      exit(1);
    }
  }
  else if(is_typecast2t(expr))
  {
  }
  else if(is_constant_int2t(expr))
  {
    //f = new Function(*vars, to_constant_int2t(expr).value.to_int64());
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

void goto_contractort::parse_intervals(irep_container<expr2t> expr)
{
  //expr->dump();
  if(is_typecast2t(expr))
    parse_intervals(to_typecast2t(expr).from);
  else if(is_and2t(expr))
  {
    parse_intervals(to_and2t(expr).side_1);
    parse_intervals(to_and2t(expr).side_2);
  }
  else if(is_lessthan2t(expr))
  {
    auto f = to_lessthan2t(expr);
    if(is_symbol2t(f.side_1))
    {
      int index =
        map->find((string)to_symbol2t(f.side_1).get_symbol_name().c_str());
      if(index != -1)
      {
        if(!is_number_type(f.side_2))
        {
          //error
        }
        auto number = to_constant_int2t(f.side_2).as_long();
        map->update_ub_interval(number, index);
      }
      else
      {
        //var is not in constraint list
        return;
      }
    }
    else if(is_symbol2t(f.side_2))
    {
      int index =
        map->find((string)to_symbol2t(f.side_2).get_symbol_name().c_str());
      if(index != -1)
      {
        if(!is_number_type(f.side_1))
        {
          //error
        }
        auto number = to_constant_int2t(f.side_1).as_long();
        map->update_ub_interval(number, index);
      }
      else
      {
        //var is not in constraint list
        return;
      }
    }
  }
  else if(is_lessthanequal2t(expr))
  {
    auto f = to_lessthanequal2t(expr);

    if(is_symbol2t(f.side_1))
    {
      int index =
        map->find((std::string)to_symbol2t(f.side_1).get_symbol_name().c_str());
      if(index != -1)
      {
        if(!is_number_type(f.side_2))
        { //error
        }
        //cout << get_expr_id(f.side_2).c_str() << endl;
        auto number = to_constant_int2t(f.side_2).as_long();
        map->update_ub_interval(number, index);
      }
      else
      {
        //error
      }
    }
    else if(is_symbol2t(f.side_2))
    {
      //look up
    }
  }
  else if(is_greaterthanequal2t(expr))
  {
    auto f = to_greaterthanequal2t(expr);
    if(is_symbol2t(f.side_1))
    {
      int index =
        map->find((string)to_symbol2t(f.side_1).get_symbol_name().c_str());
      if(index != -1)
      {
        if(!is_number_type(f.side_2))
        {
          //error
        }
        auto number = to_constant_int2t(f.side_2).as_long();
        map->update_lb_interval(number, index);
      }
      else
      {
        //ignore
      }
    }
    else if(is_symbol2t(f.side_2))
    {
      //look up
    }
  }
  else if(is_greaterthan2t(expr))
  {
  }
  else if(is_equality2t(expr))
  {
    auto f = to_equality2t(expr);
    if(is_symbol2t(f.side_1))
    {
      int index =
        map->find((string)to_symbol2t(f.side_1).get_symbol_name().c_str());
      if(index != -1)
      {
        if(!is_number_type(f.side_2))
        {
          //error
        }
        auto number = to_constant_int2t(f.side_2).as_long();
        map->update_ub_interval(number, index);
        map->update_lb_interval(number, index);
      }
      else
      {
        //ignore
      }
    }
    else if(is_symbol2t(f.side_2))
    {
      //look up
    }
  }
  else if(is_notequal2t(expr))
  {
  }
  else if(is_symbol2t(expr))
  {
    //get index;
  }
  else
  {
    //error
  }
}