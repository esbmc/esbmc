#ifndef CPROVER_GOTO_PROGRAMS_REMOVE_UNREACHABLE_H
#define CPROVER_GOTO_PROGRAMS_REMOVE_UNREACHABLE_H

#include <goto-programs/goto_functions.h>

/// @p continue_past_failed_assertions is set when claims past a violation are
/// still checked (coverage), so that an ASSERT false does not end its path.
void remove_unreachable(
  goto_functionst &goto_functions,
  bool continue_past_failed_assertions);
void remove_unreachable(
  goto_functiont &goto_function,
  bool continue_past_failed_assertions);
void remove_unreachable(
  goto_programt &goto_program,
  bool continue_past_failed_assertions);

#endif
