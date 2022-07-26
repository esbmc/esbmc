#ifndef CPROVER_GOTO_PROGRAMS_DESTRUCTOR_H
#define CPROVER_GOTO_PROGRAMS_DESTRUCTOR_H

#include <util/expr.h>
#include <util/namespace.h>

/**
 * Get an expr to represent destructor function call
 * for cpp delete
 */
code_function_callt get_destructor(const namespacet &ns, const typet &type);

#endif
