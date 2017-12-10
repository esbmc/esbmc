/*******************************************************************\

Module: Destructor Calls

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_PROGRAMS_DESTRUCTOR_H
#define CPROVER_GOTO_PROGRAMS_DESTRUCTOR_H

#include <util/namespace.h>
#include <util/std_code.h>

code_function_callt get_destructor(const namespacet &ns, const typet &type);

#endif
