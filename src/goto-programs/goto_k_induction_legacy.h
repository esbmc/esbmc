#ifndef GOTO_PROGRAMS_GOTO_K_INDUCTION_LEGACY_H_
#define GOTO_PROGRAMS_GOTO_K_INDUCTION_LEGACY_H_

#include <goto-programs/goto_functions.h>

/// --k-induction-legacy-havoc: goto_k_induction as it was before #8221.
/// Returns true when the inductive step must be disabled.
bool goto_k_induction_legacy(
  goto_functionst &goto_functions,
  bool continue_past_failed_assertions);

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_LEGACY_H_ */
