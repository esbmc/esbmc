#ifndef ESBMC_IREP2_SIMPLIFICATION_CHECK_H
#define ESBMC_IREP2_SIMPLIFICATION_CHECK_H

#include <functional>
#include <irep2/irep2_expr.h>

/** Debug-mode guard on the expression simplifier (esbmc/esbmc#4625).
 *
 *  simplify() lives in irep2, which links only fmt/bigint/immer -- not the
 *  solvers, and not util. The SMT query therefore cannot be issued from here.
 *  Instead the driver installs a checker once it holds a namespace and
 *  options, and simplify() consults whatever is installed. Nothing is
 *  installed by default, so the hook is inert in a normal run even when the
 *  build enabled it.
 *
 *  Coverage is partial by construction, and a clean run should not be read as
 *  "every rewrite proved": about ten sites call expr2t::simplify() directly
 *  rather than through the free simplify() below, and simplify() itself
 *  recurses internally, so only the outermost rewrite of each call is seen. */
namespace simplification_check
{
/** Called for every rewrite the simplifier performs. Reporting a mismatch --
 *  logging, aborting -- is the checker's business, not the hook's. */
using checkert =
  std::function<void(const expr2tc &before, const expr2tc &after)>;

void install(checkert checker);

/** Uninstall, restoring the inert default. */
void clear();

namespace detail
{
void run(const expr2tc &before, const expr2tc &after);
}

/** Consult the installed checker. Compiled out entirely unless the build was
 *  configured with -DENABLE_SIMPLIFIER_EQUIVALENCE_CHECK=ON: the simplifier
 *  runs on every expression ESBMC builds, so an unconditional indirect call
 *  would be a permanent tax on a debug aid. */
inline void verify_rewrite(const expr2tc &before, const expr2tc &after)
{
#ifdef ENABLE_SIMPLIFIER_EQUIVALENCE_CHECK
  detail::run(before, after);
#else
  (void)before;
  (void)after;
#endif
}
} // namespace simplification_check

#endif
