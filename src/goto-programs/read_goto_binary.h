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

bool read_goto_binary_array(
  const void *data,
  size_t size,
  contextt &context,
  goto_functionst &dest,
  const messaget &msg);

bool read_goto_binary(
  const std::string &path,
  contextt &context,
  goto_functionst &dest,
  const messaget &msg);

#endif
