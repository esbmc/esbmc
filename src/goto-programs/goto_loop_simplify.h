#ifndef GOTO_PROGRAMS_GOTO_LOOP_SIMPLIFY_H_
#define GOTO_PROGRAMS_GOTO_LOOP_SIMPLIFY_H_

#include <goto-programs/goto_functions.h>
#include <util/options.h>

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
/// Enabled by default. Skipped only where a loop rewrite is unsound or
/// destroys a needed signal:
///   - --termination: dead-loop erasure (Path 1) and the self-loop rewrite
///     are skipped — rewriting an infinite empty loop to assume(false) would
///     mask the non-termination verdict under test. Constant-bound counter
///     step recognition (Path 2) still runs (it only matches strictly-
///     terminating, overflow-checked loops).
///   - Coverage modes: skipped entirely (erasing loops drops branch points
///     the instrumentation needs).
/// The unwinding-assertion signal is intentionally not preserved: removing a
/// loop suppresses the "needs more unwinding" failure, which is preferable to
/// leaving the loop unverifiable.
void goto_loop_simplify(
  goto_functionst &goto_functions,
  const optionst &options);

#endif
