/// \file typecast.h
/// \brief Type casting utilities for the Solidity frontend.
///
/// Provides implicit type cast generation for Solidity expressions,
/// delegating to ESBMC's C type cast infrastructure to ensure type
/// compatibility in arithmetic and assignment contexts.

#ifndef SOLIDITY_TYPECAST_H_
#define SOLIDITY_TYPECAST_H_

#include <util/namespace.h>
#include <util/std_expr.h>

extern void
solidity_gen_typecast(const namespacet &ns, exprt &dest, const typet &type);

#endif /* SOLIDITY_TYPECAST_H_ */
