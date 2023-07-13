// This file contains the functions to replace symbolic types in
// structs, classes and unions before generating zero initializations
// or adding paddings. For more details, see PR 1180
// https://github.com/esbmc/esbmc/pull/1180

#ifndef ESBMC_SYMBOLIC_TYPES_H
#define ESBMC_SYMBOLIC_TYPES_H

#include <util/std_types.h>
#include <util/namespace.h>

// A type symbol may contain a compoment that has symbolic type,
// which doesn't work with type_byte_size.
// This function replaces the symbolic type with the complete type.
void get_complete_struct_type(struct_typet &type, const namespacet &ns);

#endif // ESBMC_SYMBOLIC_TYPES_H