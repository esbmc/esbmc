/// \file typecast.cpp
/// \brief Implementation of Solidity type casting utilities.
///
/// Wraps ESBMC's c_typecastt to perform implicit type conversions for
/// Solidity expressions, ensuring operands are cast to compatible types
/// before arithmetic, comparison, and assignment operations.

#include <solidity-frontend/typecast.h>
#include <util/c_typecast.h>
#include <util/c_types.h>
#include <util/simplify_expr_class.h>
#include <stdexcept>
#include <sstream>

void solidity_gen_typecast(const namespacet &ns, exprt &dest, const typet &type)
{
  c_typecastt c_typecast(ns);
  c_typecast.implicit_typecast(dest, type);
}
