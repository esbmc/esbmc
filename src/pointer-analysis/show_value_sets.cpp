#include <pointer-analysis/show_value_sets.h>

void show_value_sets(
  const goto_functionst &goto_functions,
  const value_set_analysist &value_set_analysis,
  std::ostream &os)
{
  value_set_analysis.output(goto_functions, os);
}

void show_value_sets(
  const goto_programt &goto_program,
  const value_set_analysist &value_set_analysis,
  std::ostream &os)
{
  value_set_analysis.output(goto_program, os);
}
