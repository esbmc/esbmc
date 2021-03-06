/*******************************************************************\

Module: Read Goto Programs

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_PROGRAMS_READ_GOTO_BINARY_H
#define CPROVER_GOTO_PROGRAMS_READ_GOTO_BINARY_H

#include <goto-programs/goto_functions.h>
#include <util/context.h>
#include <util/message/message.h>
#include <util/options.h>

void read_goto_binary(
  std::istream &in,
  contextt &context,
  goto_functionst &dest,
  const messaget &message_handler);

#endif
