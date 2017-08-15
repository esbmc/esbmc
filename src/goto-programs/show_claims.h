/*******************************************************************\

Module: Show claims

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_PROGRAMS_SHOW_CLAIMS_H
#define CPROVER_GOTO_PROGRAMS_SHOW_CLAIMS_H

#include <goto-programs/goto_functions.h>
#include <util/namespace.h>
#include <util/ui_message.h>

void show_claims(
  const namespacet &ns,
  ui_message_handlert::uit ui,
  const goto_functionst &goto_functions);

void show_claims(
  const namespacet &ns,
  ui_message_handlert::uit ui,
  const goto_programt &goto_program);

#endif
