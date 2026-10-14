#ifndef ESBMC_CLANG_CPP_FRONTEND_CLANG_CPP_DESTRUCTOR_CALL_H
#define ESBMC_CLANG_CPP_FRONTEND_CLANG_CPP_DESTRUCTOR_CALL_H

#include <util/irep/std_types.h>
#include <util/symtab/namespace.h>

/// What `delete p` calls: the destructor named statically, or the vtable slot
/// it dispatches through when the destructor is virtual. Shared because the
/// IREP2 adjust pass builds the same call (scope-clang-cpp-irep2.md \S3.14).
exprt destructor_binding(
  const namespacet &ns,
  const struct_typet &class_type,
  const struct_typet::componentt &dtor,
  const exprt &object);

#endif
