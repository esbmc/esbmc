/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_EXPR2CPP_H
#define CPROVER_EXPR2CPP_H

#include <string>

class exprt;
class namespacet;
class typet;

std::string expr2cpp(const exprt &expr, const namespacet &ns, const bool fullname = false);
std::string type2cpp(const typet &type, const namespacet &ns, const bool fullname = false);

#endif
