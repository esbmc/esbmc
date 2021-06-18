/*******************************************************************\

Module: Show Value Sets

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <pointer-analysis/show_value_sets.h>

void show_value_sets(
  const goto_functionst &goto_functions,
  const value_set_analysist &value_set_analysis)
{
  value_set_analysis.output(goto_functions, std::cout);
}

void show_value_sets(
  const goto_programt &goto_program,
  const value_set_analysist &value_set_analysis)
{
  value_set_analysis.output(goto_program, std::cout);
}
