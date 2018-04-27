/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_ARITH_TOOLS_H
#define CPROVER_ARITH_TOOLS_H

#include <util/expr.h>
#include <util/irep2.h>
#include <util/mp_arith.h>

bool to_integer(const exprt &expr, mp_integer &int_value);
exprt from_integer(const mp_integer &int_value, const typet &type);
expr2tc from_integer(const mp_integer &int_value, const type2tc &type);

mp_integer power(const mp_integer &base, const mp_integer &exponent);

// ceil(log2(size))
mp_integer address_bits(const mp_integer &size);

#endif
