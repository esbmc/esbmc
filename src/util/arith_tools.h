#ifndef CPROVER_ARITH_TOOLS_H
#define CPROVER_ARITH_TOOLS_H

#include <util/expr.h>
#include <irep2/irep2.h>
#include <util/mp_arith.h>

bool to_integer(const exprt &expr, BigInt &int_value);
exprt from_integer(const BigInt &int_value, const typet &type);
expr2tc from_integer(const BigInt &int_value, const type2tc &type);
exprt from_double(double val, const typet &type);

BigInt power(const BigInt &base, const BigInt &exponent);

// ceil(log2(size))
BigInt address_bits(const BigInt &size);

#endif
