#pragma once

#include <goto-programs/goto_functions.h>
#include <util/symtab/namespace.h>

/// Whether the user code in \p goto_functions stays inside what the
/// --partial-order prototype handles (#10990): shared globals of scalar
/// or array-of-scalar type, threads created outside loops with a null argument,
/// and pthread create, join and mutex calls. The operational models are not
/// checked. With \p verifying, also rejects what the memory model cannot
/// order yet: locks and assumptions. Logs the first construct outside that
/// scope.
bool partial_order_in_scope(
  const goto_functionst &goto_functions,
  const namespacet &ns,
  bool verifying);

/// Whether \p name is a user global whose accesses --partial-order records
/// as events. The operational model's own state is not one.
bool is_partial_order_global(const namespacet &ns, const irep_idt &name);

/// The types of the globals --partial-order handles: scalars and arrays of
/// them. An array is one object; an assignment to an element updates it.
bool is_partial_order_type(const type2tc &t);
