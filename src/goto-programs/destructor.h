#ifndef CPROVER_GOTO_PROGRAMS_DESTRUCTOR_H
#define CPROVER_GOTO_PROGRAMS_DESTRUCTOR_H

#include <util/namespace.h>
#include <util/std_code.h>

bool get_destructor(
  const namespacet &ns,
  const typet &type,
  code_function_callt &destructor);

#endif
