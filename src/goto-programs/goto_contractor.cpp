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
    //std::cout << "hello " << it->second.body_available << "\n";
    if(it->second.body_available)
    {
      std::cout << it->first << "\n";
      //std::cout << it->second.body.instructions.max_size() << "\n";
      goto_contractort(it->first, goto_functions, it->second, message_handler);
    }
  }
  goto_functions.update();
}

void goto_contractort::get_intervals(goto_functionst goto_functions)
{
  Forall_goto_functions(it, goto_functions)
    if(it->second.body_available)
    {
    }
  auto function = goto_functions.function_map.find("c:@F@main");
  std::cout << "found main: " << function->first << "\n";
  auto it = function->second.body.instructions.begin();
  while(it != function->second.body.instructions.end())
  {
    if(it->is_assume())
    {
      std::cout << "found assume at:" << it->location.as_string() << " : "
                << it->location_number << "\n"; // <<  <<"\n";
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
  auto loop_head = loop.get_original_loop_head();
  loop_head->dump();
  auto loop_exit = loop.get_original_loop_exit();
  loop_exit->dump();

  goto_programt dest(message_handler);

  expr2tc zero = gen_zero(get_uint32_type());
  expr2tc one = gen_one(get_uint32_type());
  greaterthanequal2tc cond(one, zero);

  std::cout << "test to string for intervals" << cond <<std::endl;
  cond->dump();
  simplify(cond);
  cond->dump();

  assume_cond(cond, dest, loop_exit->location);
  dest.dump();
  goto_function.body.insert_swap(loop_exit,dest);
}

// copied from k-induction
