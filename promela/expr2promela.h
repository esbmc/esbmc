/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_EXPR2PROMELA_H
#define CPROVER_EXPR2PROMELA_H

#include <ansi-c/expr2c.h>

std::string expr2promela(const exprt &expr, const namespacet &ns);
std::string type2promela(const typet &type, const namespacet &ns);

#endif
