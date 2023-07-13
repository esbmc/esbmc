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

// A type symbol may contain a compoment that has symbolic type,
// which doesn't work with type_byte_size.
// This function replace the symbolic type with the actual type.
void get_complete_struct_type(struct_typet &type, const namespacet &ns);

#endif // CPROVER_ANSI_C_PADDING_H
