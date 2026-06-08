#ifndef CPROVER_ARITH_TOOLS_H
#define CPROVER_ARITH_TOOLS_H

#include <util/expr.h>
#include <irep2/irep2.h>
#include <util/mp_arith.h>

bool to_integer(const exprt &expr, BigInt &int_value);

/// IREP2 twin of to_integer(const exprt&): fold @p expr to an integer value,
/// returning false on success (value written to @p int_value) and true on
/// failure (not a foldable integer constant). Mirrors the legacy contract.
/// A constant_int / constant_bool yields its value directly; a typecast of a
/// constant is folded through the simplifier so the cast (truncation, sign
/// change, bool conversion) is applied — never extract the operand's raw value
/// across a cast. Callers that have already simplified @p expr hit only the
/// bare-constant paths.
bool to_integer(const expr2tc &expr, BigInt &int_value);

exprt from_integer(const BigInt &int_value, const typet &type);
expr2tc from_integer(const BigInt &int_value, const type2tc &type);
exprt from_double(double val, const typet &type);

BigInt power(const BigInt &base, const BigInt &exponent);

// ceil(log2(size))
BigInt address_bits(const BigInt &size);

#endif
