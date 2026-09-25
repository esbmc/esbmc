#ifndef CPROVER_GOTO_PROGRAMS_REMOVE_NO_OP_H
#define CPROVER_GOTO_PROGRAMS_REMOVE_NO_OP_H

#include <goto-programs/goto_functions.h>

bool is_no_op(
  const goto_programt &,
  goto_programt::const_targett,
  bool ignore_labels = false);
/// True when every instruction of @p p is a no-op, i.e. the program is
/// observationally empty. generate_ifthenelse and the native if-arm both branch
/// on this, so they must agree on it.
bool is_no_op_program(const goto_programt &p);
void remove_no_op(goto_programt &goto_program);
void remove_no_op(goto_functionst &goto_functions);

#endif
