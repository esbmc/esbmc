#pragma once

#include <goto-programs/goto_functions.h>
#include <util/symtab/namespace.h>

#include <map>
#include <vector>

/// Whether the user code in \p goto_functions stays inside what the
/// --partial-order prototype handles (#10990): shared globals that are
/// scalars, pointers to globals, or arrays and structs of scalars, threads
/// created with a null argument, and
/// pthread create, join, exit, mutex and condition-variable calls. The
/// operational models are not checked. With \p races, also rejects global
/// arrays. With \p deadlocks, also rejects pthread_exit, and, with
/// \p verifying, assumptions, which --deadlock-check keeps on the whole
/// equation whatever the clocks say.
/// Logs the first construct outside that scope.
bool partial_order_in_scope(
  const goto_functionst &goto_functions,
  const namespacet &ns,
  bool verifying,
  bool races,
  bool deadlocks);

/// Whether \p name is a user global whose accesses --partial-order records
/// as events. The operational model's own state is not one.
bool is_partial_order_global(const namespacet &ns, const irep_idt &name);

/// The types of the globals --partial-order handles: scalars, and arrays and
/// structs of them. An array or a struct is one object; an assignment to an
/// element or a field updates it.
bool is_partial_order_type(const type2tc &t);

/// is_partial_order_type, or a pointer. A global pointer is a scalar whose
/// values the scope check limits to globals' addresses and null.
bool is_partial_order_global_type(const type2tc &t);

/// Every value assigned to each shared global pointer, its initialiser
/// included, by the pointer's name.
std::map<irep_idt, std::vector<expr2tc>> partial_order_pointer_values(
  const goto_functionst &goto_functions,
  const namespacet &ns);
