// This file contains the functions to replace symbolic types in
// structs, classes and unions before generating zero initializations
// or adding paddings. For more details, see PR 1180
// https://github.com/esbmc/esbmc/pull/1180

#ifndef ESBMC_SYMBOLIC_TYPES_H
#define ESBMC_SYMBOLIC_TYPES_H

#include <util/std_types.h>
#include <util/namespace.h>

// Replaces the symbolic struct type with the complete struct type.
void get_complete_struct_type(struct_typet &type, const namespacet &ns);

// If an array type's subtype is symbolic struct type, replace it with
// the complete struct type
bool array_type_contains_symbolic(const typet &type,
    typet &complete_type, const namespacet &ns);

#endif // ESBMC_SYMBOLIC_TYPES_H