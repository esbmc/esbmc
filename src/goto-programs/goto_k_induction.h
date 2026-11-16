#ifndef GOTO_PROGRAMS_GOTO_K_INDUCTION_H_
#define GOTO_PROGRAMS_GOTO_K_INDUCTION_H_

#include <goto-programs/goto_functions.h>
#include <util/symtab/context.h>

/// Per-loop k-induction transformation: havoc each loop's modified
/// variables and inject an ASSUME of the loop entry condition right
/// before the loop head. With @p context, a state that fails the loop's head
/// test skips the loop, so it is neither havoced nor constrained (#10154); the
/// flag that records this is a new symbol there. Without it every loop is
/// assumed to be entered, which --termination needs: non-termination is a
/// claim about the states that enter the loop. Applied to every function with a
/// body (including __ESBMC_main, which is loop-free and therefore untouched,
/// and library helpers).
///
/// A loop that writes through a pointer also havocs that storage: `*p` while
/// the loop leaves `p` alone, otherwise the named objects the whole-program
/// Andersen analysis resolves the pointer to. Returns true, after logging why,
/// when the inductive step would be unsound: a write in a reachable user
/// function can be covered neither way (issues #5224, #5230), a reachable call
/// to __VERIFIER_nondet_memory (#5593), or a loop that allocates under a leak
/// check. The caller must then disable the inductive step.
/// @p continue_past_failed_assertions is set when claims past a violation are
/// still checked (--multi-property, coverage), so that a path through a failed
/// assertion still counts as entering the loop (#7900).
bool goto_k_induction(
  goto_functionst &goto_functions,
  contextt *context,
  bool continue_past_failed_assertions);

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_H_ */
