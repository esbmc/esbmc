#ifndef _ESBMC_PROP_SMT_SIMPLIFICATION_EQUIVALENCE_H_
#define _ESBMC_PROP_SMT_SIMPLIFICATION_EQUIVALENCE_H_

#include <atomic>
#include <irep2/irep2_expr.h>
#include <util/symtab/namespace.h>
#include <util/config/options.h>

/** Verdict on one simplifier rewrite (esbmc/esbmc#4625). */
enum class simplification_equivalencet
{
  /** The solver proved before == after for every valuation. */
  equivalent,
  /** The solver found a valuation where they differ: a simplifier bug. */
  differs,
  /** Not decided -- a shape the check declines, or a solver failure. */
  skipped
};

/** Ask an SMT solver whether a simplifier rewrite preserved meaning.
 *
 *  Free symbols stay free, so `equivalent` means equivalence under every
 *  valuation, not merely on some model. Declines (returns `skipped`) rather
 *  than guessing for shapes whose equality is not a plain SMT question --
 *  pointers, side effects, code, and anything the conversion rejects. */
simplification_equivalencet check_simplification_equivalence(
  const expr2tc &before,
  const expr2tc &after,
  const namespacet &ns,
  const optionst &options);

/** Install the above as simplify()'s checker for the rest of the run: every
 *  rewrite is proved, and a `differs` verdict logs both expressions and exits.
 *  Does nothing unless the build enabled the check. */
void install_simplification_equivalence_check(
  const namespacet &ns,
  const optionst &options);

/** How much the installed checker actually decided. Without this a run that
 *  declined every rewrite is indistinguishable from one that proved them all,
 *  which is the failure mode that would make the whole check worthless. */
namespace simplification_check_stats
{
extern std::atomic<unsigned long> proved;
extern std::atomic<unsigned long> declined;

void report();
} // namespace simplification_check_stats

#endif
