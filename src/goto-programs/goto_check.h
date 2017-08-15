/*******************************************************************\

Module: Program Transformation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_PROGRAMS_GOTO_CHECK_H
#define CPROVER_GOTO_PROGRAMS_GOTO_CHECK_H

#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_program.h>
#include <util/namespace.h>
#include <util/options.h>

void goto_check(
  const namespacet &ns,
  optionst &options,
  goto_functionst &goto_functions);

#endif
