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
    {
      //std::cout << "found assert: \n";
      //it->dump();
      constraint = create_constraint_from_expr2t(it->guard);
      //cout << "here is the constraint" << *constraint << endl;
    }
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
    {
      //std::cout << "found assume at:" << it->location.as_string() << " : "
      //         << it->location_number << "\n";
      //it->guard->dump();
      parse_intervals(it->guard);
      //map->dump();
    }
    it++;
  }
  //std::cout << "------------------get vars" << std::endl;
  //only last loop
  /*loopst loop;
  for(auto &function_loop : function_loops)
    loop = function_loop;

  //found vars as string.
  auto vars_ESBMC = loop.get_modified_loop_vars();
  string vars_CSP[vars_ESBMC.size()];
  int i = 0;
  for(const auto &item : vars_ESBMC)
    if(is_symbol2t(item))
      vars_CSP[i++] = to_symbol2t(item).get_symbol_name();*/
}

void goto_contractort::insert_assume(
  goto_functionst goto_functions,
  IntervalVector new_intervals)
{
  //auto function = goto_functions.function_map.find("c:@F@main");

  std::cout << "------------------loop" << std::endl;
  loopst loop;
  for(auto &function_loop : function_loops)
    loop = function_loop;

  auto loop_head = loop.get_original_loop_head();
  loop_head->dump();
  auto loop_exit = loop.get_original_loop_exit();
  loop_exit->dump();

  goto_programt dest(message_handler);

  expr2tc zero = gen_zero(get_uint32_type());
  expr2tc one = gen_one(get_uint32_type());
  greaterthanequal2tc cond(one, zero);
  for(int i = 0; i < map->size(); i++)
  {

    std::cout << "test to string for intervals" << cond << std::endl;
    cond->dump();
    simplify(cond);
    cond->dump();

    assume_cond(cond, dest, loop_exit->location);
    dest.dump();
    goto_function.body.insert_swap(loop_exit, dest);
  }
}

IntervalVector goto_contractort::contractor()
{
  //NumConstraint c((char *)vars[0], (char *)vars[1], (char *)constraint);
  NumConstraint c = *constraint;
  std::cout << "My Constraint:" << c << std::endl;
  std::cout << "My Function:" << c.f << std::endl;

  auto complement = LEQ;
  if(!c.right_hand_side().lb().is_zero())
    complement = GEQ; //wrong

  NumConstraint c2(c.f, complement);
  std::cout << "My complement Constraint:" << c2 << std::endl;
  std::cout << "My Function:" << c2.f << std::endl;

  CtcFwdBwd c_out(c);
  CtcFwdBwd c_in(c2);

  domains = map->intervals;
  std::cout << "My domains:" << domains << std::endl;
  auto X = domains;
  /*c_out.contract(X);
  IntervalVector *s_out;
  int num = domains.diff(X, s_out);
  std::cout << "My domains after outer contractor:" << X << std::endl;
  for(int i = 0; i < num; i++)
    std::cout << "s_out[" << i << "]:" << s_out[i] << std::endl;*/

  //X = domains;
  IntervalVector *s_in;
  c_in.contract(X);
  int num = domains.diff(X, s_in);
  std::cout << "My domains after Inner contractor:" << X << std::endl;
  for(int i = 0; i < num; i++)
    std::cout << "s_in[" << i << "]:" << s_in[i] << std::endl;

  return X;
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
  //>,<,<=
  return nullptr;
}
//+-
Function *
goto_contractort::create_function_from_expr2t(irep_container<expr2t> expr)
{
  //auto op = get_expr_id(expr);
  Function *f = nullptr;
  Function *g, *h;
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
  return f;
}

int goto_contractort::create_variable_from_expr2t(irep_container<expr2t> expr)
{
  std::string var_name = to_symbol2t(expr).get_symbol_name().c_str();
  int index = map->find(var_name);
  if(index == -1)
  {
    int new_index = map->add_var(var_name);
    map->dump();
    if(new_index != -1)
      return new_index;
    return -1;
  }
  return index;
}

void goto_contractort::parse_intervals(irep_container<expr2t> expr)
{
  if(is_lessthan2t(expr))
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
        //cout<<get_expr_id(f.side_2).c_str()<<endl;
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
  }
  else if(is_greaterthan2t(expr))
  {
  }
  else if(is_equality2t(expr))
  {
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