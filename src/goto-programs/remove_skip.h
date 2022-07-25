#ifndef CPROVER_GOTO_PROGRAMS_REMOVE_SKIP_H
#define CPROVER_GOTO_PROGRAMS_REMOVE_SKIP_H

#include <goto-programs/goto_functions.h>

bool is_skip(
  const goto_programt &,
  goto_programt::const_targett,
  bool ignore_labels = false);
void remove_skip(goto_programt &goto_program);
void remove_skip(goto_functionst &goto_functions);

#endif
