#ifndef CPROVER_GOTO_PROGRAMS_SHOW_VALUE_SETS_H
#define CPROVER_GOTO_PROGRAMS_SHOW_VALUE_SETS_H

#include <goto-programs/goto_functions.h>
#include <pointer-analysis/value_set_analysis.h>
#include <util/namespace.h>

void show_value_sets(
  const goto_functionst &goto_functions,
  const value_set_analysist &value_set_analysis,
  std::ostream &os);

void show_value_sets(
  const goto_programt &goto_program,
  const value_set_analysist &value_set_analysis,
  std::ostream &os);

#endif
