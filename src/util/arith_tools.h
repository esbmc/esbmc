/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_ARITH_TOOLS_H
#define CPROVER_ARITH_TOOLS_H

#include <util/expr.h>
#include <util/mp_arith.h>

bool to_integer(const exprt &expr, mp_integer &int_value);
exprt from_integer(const mp_integer &int_value, const typet &type);

mp_integer power(const mp_integer &base, const mp_integer &exponent);

#endif
