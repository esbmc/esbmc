/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_SIMPLIFY_EXPR_H
#define CPROVER_SIMPLIFY_EXPR_H

#include <expr.h>

//
// simplify an expression
//
// true: did nothing
// false: simplified something
//

bool simplify(exprt &expr);

#endif
