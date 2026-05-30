#include <esbmc/non_termination.h>

/// Non-termination check entry point — currently a no-op.
///
/// A prototype "trivial-recurrent-set" detector (walk every
/// `while(1)`-shaped loop looking for a non-exiting path that writes no
/// live-across-iterations symbol) was implemented and measured on the
/// SV-COMP termination set. Result:
///   * 0 wins on the eca-rers2012 cluster (200 expected-false benchmarks):
///     every eca update path modifies some global state-machine
///     variable, so the "no live writes" predicate is never satisfied.
///   * Net negative on the ldv-linux-3.4-simple "infinite_withcheck_stateful"
///     cluster (200 sampled): 68 correct refutations, 16 wrong-false
///     verdicts (SV-COMP expected_verdict: true). The wrong-falses
///     reflect benchmarks where the loop is structurally non-terminating
///     under the demon's nondeterministic choices but the SV-COMP label
///     asserts termination — likely an upstream labelling decision tied
///     to a semantic property the detector cannot recover syntactically.
///     At -16 per wrong-false vs +1 per right-false the cluster loses 188
///     raw, dominating any other potential gains.
///
/// The prototype's helpers (live-on-entry, clean-non-exit DFS) are kept
/// in version control for reference; reintroducing them requires either
/// (a) a syntactic feature that separates the mislabelled subset or
/// (b) an SMT discharge that confirms the demon really CAN keep the loop
/// alive on the candidate path (e.g. proving its branch system
/// satisfiable across iterations). Until then this entry point returns
/// UNKNOWN unconditionally and the existing ranking / k-induction
/// machinery owns every termination verdict.
tvt try_prove_non_termination_by_recurrent_set(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns)
{
  (void)goto_functions;
  (void)options;
  (void)ns;
  return tvt(tvt::TV_UNKNOWN);
}
