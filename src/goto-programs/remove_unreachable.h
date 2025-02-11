#ifndef CPROVER_GOTO_PROGRAMS_REMOVE_UNREACHABLE_H
#define CPROVER_GOTO_PROGRAMS_REMOVE_UNREACHABLE_H

#include <goto-programs/goto_functions.h>

void remove_unreachable(goto_functionst &goto_functions);
void remove_unreachable(goto_functiont &goto_function);
void remove_unreachable(goto_programt &goto_program);

#endif
