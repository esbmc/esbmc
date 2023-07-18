/// \file
/// ANSI-C Language Type Checking

#ifndef CPROVER_ANSI_C_PADDING_H
#define CPROVER_ANSI_C_PADDING_H

#include <util/std_types.h>
#include <util/namespace.h>
#include <util/mp_arith.h>

BigInt alignment(const typet &type, const namespacet &);
void add_padding(struct_typet &type, const namespacet &);
void add_padding(union_typet &type, const namespacet &);

#endif // CPROVER_ANSI_C_PADDING_H
