/*******************************************************************\

Module: Race Detection for Threaded Goto Programs

Author: Daniel Kroening

Date: February 2006

\*******************************************************************/

#ifndef CPROVER_GOTO_PROGRAMS_RACE_DETECTION_H
#define CPROVER_GOTO_PROGRAMS_RACE_DETECTION_H

#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_program.h>
#include <pointer-analysis/value_sets.h>

void add_race_assertions(
  value_setst &value_sets,
  contextt &context,
  goto_programt &goto_program);

void add_race_assertions(
  value_setst &value_sets,
  contextt &context,
  goto_functionst &goto_functions);

#endif
