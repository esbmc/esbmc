#ifndef _ESBMC_PROP_SMT_SIMPLIFICATION_EQUIVALENCE_H_
#define _ESBMC_PROP_SMT_SIMPLIFICATION_EQUIVALENCE_H_

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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
  /** `after` has operands whose sorts disagree: also a simplifier bug, and one
   *  that would abort any backend rather than yield a wrong answer. */
  malformed,
  /** Not decided -- a shape the check declines, or a solver failure. */
  skipped
};

class smt_convt;

/** Asks an SMT solver whether simplifier rewrites preserved meaning.
 *
 *  Holds one solver per calling thread for its lifetime and pushes a frame per
 *  query. A solver per query would be simpler, but create_solver leaks its
 *  tuple flattener (solve.cpp) -- ~14 KB, harmless at one call per run and
 *  fatal at one call per rewrite, which is what the installed checker does.
 *  Per thread rather than one flat: --parallel-solving simplifies on several
 *  threads at once, and an smt_convt is not safe to share between them.
 *
 *  Free symbols stay free, so `equivalent` means equivalence under every
 *  valuation, not merely on some model. Declines (returns `skipped`) rather
 *  than guessing for shapes whose equality is not a plain SMT question --
 *  pointers, side effects, code, and terms conversion cannot be asked to
 *  build -- the last of which has to be screened up front, see sorts_agree()
 *  in the implementation (esbmc/esbmc#7220). */
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
  /** Format the live model's valuation of `before`/`after`'s free symbols into
   *  @p witness. Only callable while the frame that produced the model is
   *  still pushed. */
  void record_witness(
    const expr2tc &before,
    const expr2tc &after,
    smt_convt &ctx,
    std::string &witness);

  /** This thread's solver, built on first use. */
  smt_convt &solver();
  /** Drop this thread's solver: after an exception its state is not
   *  trustworthy, so the next check() builds a fresh one. */
  void drop_solver();

  namespacet ns;
  optionst options;
  std::mutex solvers_mutex;
  std::map<std::thread::id, std::unique_ptr<smt_convt>> solvers;
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

/** Reports the stats and uninstalls the checker when it goes out of scope.
 *  doit() leaves down twenty different paths and the checker captured a
 *  namespace over a member of it, so tying both to a scope is what keeps the
 *  count covering the whole run -- symex included -- without the namespace
 *  outliving what it points at (esbmc/esbmc#7260). */
struct simplification_check_scopet
{
  ~simplification_check_scopet();
};

/** How much check() actually decided, over every caller. Without this a run
 *  that declined every rewrite is indistinguishable from one that proved them
 *  all, which is the failure mode that would make the whole check worthless. */
namespace simplification_check_stats
{
extern std::atomic<unsigned long> proved;
extern std::atomic<unsigned long> declined;
/** Of the declines, those whose `before` was already ill-sorted -- a defect
 *  elsewhere in ESBMC rather than a limit of the check, so a nonzero count is
 *  something to go and look at. */
extern std::atomic<unsigned long> ill_sorted;
/** Rewrites the check rejected. Reported as they are found rather than on the
 *  first one, so a run surfaces the whole set (esbmc/esbmc#7326). */
extern std::atomic<unsigned long> violations;

void report();
} // namespace simplification_check_stats

#endif
