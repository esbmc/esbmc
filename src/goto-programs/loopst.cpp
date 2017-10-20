/*
 * loopst.cpp
 *
 *  Created on: Jul 20, 2015
 *      Author: mramalho
 */

#include <goto-programs/loopst.h>

const loopst::loop_varst &loopst::get_loop_vars() const
{
  return loop_vars;
}

goto_programt::targett loopst::get_original_loop_exit() const
{
  return original_loop_exit;
}

void loopst::set_original_loop_exit(goto_programt::targett _loop_exit)
{
  original_loop_exit = _loop_exit;
}

goto_programt::targett loopst::get_original_loop_head() const
{
  return original_loop_head;
}

void loopst::set_original_loop_head(goto_programt::targett _loop_head)
{
  original_loop_head = _loop_head;
}

void loopst::add_var_to_loop(loopst::loop_varst &_loop_vars)
{
  loop_vars.insert(_loop_vars.begin(), _loop_vars.end());
}

void loopst::add_var_to_loop(const expr2tc &expr)
{
  loop_vars.insert(expr);
}

void loopst::dump()
{
  unsigned n=original_loop_head->location_number;

  std::cout << n << " is head of (size: ";
  std::cout << size;
  std::cout << ") { ";

  for(goto_programt::instructionst::iterator l_it = original_loop_head;
      l_it != original_loop_exit;
      ++l_it)
    std::cout << (*l_it).location_number << ", ";
  std::cout << original_loop_exit->location_number;

  std::cout << " }" << std::endl;

  dump_loop_vars();
}

void loopst::dump_loop_vars()
{
  std::cout << "Loop variables:\n";
  unsigned int i = 0;
  for (auto var : loop_vars)
    std::cout << ++i << ". \t" << to_symbol2t(var).thename << '\n';
  std::cout << '\n';
}
