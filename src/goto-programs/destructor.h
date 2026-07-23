#ifndef CPROVER_GOTO_PROGRAMS_DESTRUCTOR_H
#define CPROVER_GOTO_PROGRAMS_DESTRUCTOR_H

#include <util/namespace.h>
#include <util/std_code.h>
#include <util/std_types.h>

/// Follows \p type to the authoritative class type symbol, or nullptr when it
/// does not resolve to a struct.
///
/// The result points either into the symbol table or, for a tag that is not
/// bound, into \p type itself, so it stays valid only as long as both do:
/// callers must not pass a temporary, nor erase symbols before using it.
const struct_typet *resolve_class_type(const namespacet &ns, const typet &type);

/// The destructor method component of \p struct_type, or nullptr when it has
/// none. The result points into \p struct_type and shares its lifetime.
const struct_typet::componentt *
get_destructor_component(const namespacet &ns, const struct_typet &struct_type);

bool get_destructor(
  const namespacet &ns,
  const typet &type,
  code_function_callt &destructor);

#endif
