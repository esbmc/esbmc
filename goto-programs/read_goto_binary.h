/*******************************************************************\

Module: Read Goto Programs

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_PROGRAMS_READ_GOTO_BINARY_H
#define CPROVER_GOTO_PROGRAMS_READ_GOTO_BINARY_H

#include <message.h>
#include <options.h>
#include <context.h>

#include "goto_functions.h"

void read_goto_binary(
  std::istream &in,
  contextt &context,
  goto_functionst &dest,
  message_handlert &message_handler);

#endif
