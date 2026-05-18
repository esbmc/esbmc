#ifndef GOTO_PROGRAMS_GOTO_LOOP_SIMPLIFY_H_
#define GOTO_PROGRAMS_GOTO_LOOP_SIMPLIFY_H_

#include <goto-programs/goto_functions.h>

/// Detect and remove no-op loops at goto-program level, before symex.
///
/// A loop is removable when:
///   - The body modifies only non-pointer local symbols.
///   - The body contains no FUNCTION_CALL / ASSERT / ASSUME / non-back-edge GOTO.
///   - All modified symbols are dead immediately after the loop (recognised
///     via DEAD instructions emitted by the C frontend), so the loop has no
///     observable effect on post-loop state.
///
/// When all conditions hold the entire loop (head IF, body, back-edge) is
/// rewritten to SKIP. A subsequent remove_no_op call cleans the SKIPs.
///
/// Iterates to a fixed point so nested no-op loops collapse outward.
///
/// Skipped under --termination or --unwinding-assertions, where loop
/// presence is itself observable.
void goto_loop_simplify(goto_functionst &goto_functions);

#endif
