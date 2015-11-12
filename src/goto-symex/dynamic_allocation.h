/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_DYNAMIC_ALLOCATION_H
#define CPROVER_DYNAMIC_ALLOCATION_H

#include <irep2.h>
#include <namespace.h>

void default_replace_dynamic_allocation(
  const namespacet &ns,
  expr2tc &expr);

#endif
