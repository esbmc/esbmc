//
// Created by Mohannad Aldughaim on 09/01/2022.
//

#include "goto_contractor.h"

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

void goto_contractort::get_intervals(goto_functionst goto_functions)
{
  auto function = goto_functions.function_map.find("c:@F@main");
  std::cout << "found main: " << function->first << "\n";
  auto it = function->second.body.instructions.begin();
  while(it != function->second.body.instructions.end())
  {
    if(it->is_assume())
    {
      std::cout << "found assume at:" << it->location.as_string() << " : "
                << it->location_number << "\n";
      it->dump();
    }
    else if(it->is_assert())
    {
      std::cout << "found assert at:" << it->location.as_string() << " : "
                << it->location_number << "\n";
      it->dump();
    }
    it++;
  }
}

void goto_contractort::insert_assume(goto_functionst goto_functions)
{
  auto function = goto_functions.function_map.find("c:@F@main");

  std::cout << "------------------loop" << std::endl;
  loopst loop;
  for(auto &function_loop : function_loops)
    loop = function_loop;

  loop.dump();
  //TODO: get vars.
  //vars = loop.get_modified_loop_vars();
  auto loop_head = loop.get_original_loop_head();
  loop_head->dump();
  auto loop_exit = loop.get_original_loop_exit();
  loop_exit->dump();

  goto_programt dest(message_handler);


  expr2tc zero = gen_zero(get_uint32_type());
  expr2tc one = gen_one(get_uint32_type());
  greaterthanequal2tc cond(one, zero);

  std::cout << "test to string for intervals" << cond << std::endl;
  cond->dump();
  simplify(cond);
  cond->dump();

  assume_cond(cond, dest, loop_exit->location);
  dest.dump();
  goto_function.body.insert_swap(loop_exit, dest);
}

IntervalVector goto_contractort::contractor(
  int n_vars,
  string **vars,
  IntervalVector domains,
  string *constraint)
{
  //Interval x(0, INFINITY);
  //Interval y(0, 1000);
  //IntervalVector domains = {x, y};
  if(n_vars != 2)
  {
    exit(1);
  }
  //domains[0].lb();
  //domains[1].ub();
  NumConstraint c((char*)vars[0], (char*)vars[1], (char *)constraint);
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