#ifndef _ESBMC_PROP_SMT_SIMPLIFICATION_EQUIVALENCE_H_
#define _ESBMC_PROP_SMT_SIMPLIFICATION_EQUIVALENCE_H_

#include <atomic>
#include <memory>
#include <string>
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

class smt_convt;

/** Asks an SMT solver whether simplifier rewrites preserved meaning.
 *
 *  Holds one solver for its lifetime and pushes a frame per query. A solver
 *  per query would be simpler, but create_solver leaks its tuple flattener
 *  (solve.cpp) -- ~14 KB, harmless at one call per run and fatal at one call
 *  per rewrite, which is what the installed checker does.
 *
 *  Free symbols stay free, so `equivalent` means equivalence under every
 *  valuation, not merely on some model. Declines (returns `skipped`) rather
 *  than guessing for shapes whose equality is not a plain SMT question --
 *  pointers, side effects, code, and anything the conversion rejects. */
class simplification_equivalence_checkert
{
public:
  simplification_equivalence_checkert(
    const namespacet &ns,
    const optionst &options);
  ~simplification_equivalence_checkert();

  /** @param witness when non-null and the verdict is `differs`, receives a
   *         valuation of the free symbols on which the two disagree. */
  simplification_equivalencet check(
    const expr2tc &before,
    const expr2tc &after,
    std::string *witness = nullptr);

private:
  namespacet ns;
  optionst options;
  std::unique_ptr<smt_convt> ctx;
};

/** One-shot convenience over the above; builds and discards a solver. */
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
