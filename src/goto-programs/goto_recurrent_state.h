#ifndef GOTO_PROGRAMS_GOTO_RECURRENT_STATE_H_
#define GOTO_PROGRAMS_GOTO_RECURRENT_STATE_H_

#include <goto-programs/goto_functions.h>
#include <util/symtab/context.h>
#include <vector>

/// A loop whose head goto_recurrent_state instrumented.
struct recurrent_state_loopt
{
  locationt head;
  /// Fails when a state of the loop comes back at its head.
  goto_programt::targett check;
};

/// The non-termination check of Metta et al. (arXiv:2409.12985): at the head
/// of each loop, save on a nondet visit what the loop writes, and assert that
/// no later visit sees it again. A failed check is a reachable state the loop
/// returns to, so the loop can run forever.
///
/// Rewrites every other ASSERT into an ASSUME, so a base case over the result
/// fails only on a recurrent state. Leaves alone a loop whose writes it cannot
/// name: one through a pointer, a function pointer or recursion.
std::vector<recurrent_state_loopt>
goto_recurrent_state(goto_functionst &goto_functions, contextt &context);

#endif /* GOTO_PROGRAMS_GOTO_RECURRENT_STATE_H_ */
