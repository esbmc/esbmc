#ifndef GOTO_PROGRAMS_GOTO_VALUE_SET_INVARIANT_H_
#define GOTO_PROGRAMS_GOTO_VALUE_SET_INVARIANT_H_

#include <goto-programs/goto_functions.h>
#include <pointer-analysis/value_set_analysis.h>
#include <util/namespace.h>

/// For each loop that contains modified pointer variables, synthesise a
/// loop invariant of the form (p == &a || p == &b || ...) from the
/// static value-set analysis and inject it as a LOOP_INVARIANT instruction
/// immediately before the loop head.
///
/// The existing goto_loop_invariant_combined pass then processes these
/// injected instructions: it creates Branch 1 (base-case ASSERT + HAVOC +
/// ASSUME + body copy + inductive-step ASSERT) and inserts ASSUME(INV) at
/// the end of the original loop body so that goto_k_induction can exploit it.
void goto_inject_value_set_invariants(
  goto_functionst &goto_functions,
  value_set_analysist &vsa,
  const namespacet &ns);

#endif /* GOTO_PROGRAMS_GOTO_VALUE_SET_INVARIANT_H_ */
