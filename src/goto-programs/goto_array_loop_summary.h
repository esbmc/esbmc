#ifndef GOTO_PROGRAMS_GOTO_ARRAY_LOOP_SUMMARY_H_
#define GOTO_PROGRAMS_GOTO_ARRAY_LOOP_SUMMARY_H_

#include <goto-programs/goto_functions.h>
#include <util/config/options.h>
#include <util/symtab/context.h>

/// Replace a counting loop that writes only `a[i]` with its effect, so the
/// loop needs no unwinding:
///
///   for (; i < hi; i++) { [if (g(i))] a[i] = f(i); }
///
/// becomes one iteration at a nondet index in [i, hi), on that element alone,
/// so every check in the body still fires for each index (left out when the
/// body can fail no check), then
///
///   a = lambda j. (i <= j < hi && g(j)) ? f(j) : a[j];  i = hi;
///
/// A loop whose body only asserts, directly or by calling __VERIFIER_assert,
/// is the same rewrite without the write. Runs before --goto-unwind,
/// goto_check and the k-induction passes.
void goto_array_loop_summary(
  goto_functionst &goto_functions,
  contextt &context,
  const optionst &options);

#endif
