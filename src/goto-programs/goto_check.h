#ifndef CPROVER_GOTO_PROGRAMS_GOTO_CHECK_H
#define CPROVER_GOTO_PROGRAMS_GOTO_CHECK_H

#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_program.h>
#include <util/namespace.h>
#include <util/options.h>

// to invoke "get_base_object" for input overflow checks
#include <util/type_byte_size.h>

void goto_check(
  const namespacet &ns,
  optionst &options,
  goto_functionst &goto_functions);

#endif
