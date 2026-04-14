#ifndef GOTO_PROGRAMS_GOTO_ATOMICITY_CHECK_H_
#define GOTO_PROGRAMS_GOTO_ATOMICITY_CHECK_H_

#include <goto-programs/goto_functions.h>
#include <util/context.h>
#include <util/namespace.h>

/// Post-GOTO-conversion atomicity check instrumentation pass.
///
/// For each assignment to or read from a global variable, inserts:
///   - Snapshot assignments: tmp_i = global_i
///   - If the LHS is also a global: ATOMIC_BEGIN ... ATOMIC_END wrapping
///   - An assertion: assert(tmp_i == global_i) checking for concurrent
///     modification between the snapshot and the assignment.
///
/// This corresponds to the --atomicity-check flag and detects violations of
/// atomicity when multiple threads access shared global variables.
void goto_atomicity_check(
  goto_functionst &goto_functions,
  const namespacet &ns,
  contextt &context);

#endif // GOTO_PROGRAMS_GOTO_ATOMICITY_CHECK_H_
