#ifndef CPROVER_GOTO_PROGRAMS_WRITE_GOTO_BINARY_H_
#define CPROVER_GOTO_PROGRAMS_WRITE_GOTO_BINARY_H_

#define GOTO_BINARY_VERSION 1

#include <goto-programs/goto_functions.h>
#include <ostream>
#include <util/context.h>

bool write_goto_binary(
  std::ostream &out,
  const contextt &lcontext,
  goto_functionst &functions);

#endif
