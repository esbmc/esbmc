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
 *  The hook is consulted per node, at the point each rewrite is made, rather
 *  than once per outermost call (esbmc/esbmc#7260): a peephole is what changes
 *  meaning, and checking whole expressions saw neither the peepholes nor the
 *  direct expr2t::simplify() callers that never reach the free simplify()
 *  below. Coverage is still not total, and three gaps are structural:
 *  simplify() returns nil for address_of and overflow before any rewrite site;
 *  the driver cannot install a checker until it holds a namespace, so frontend
 *  parsing runs unchecked; and the rewrites the checker's own solver performs
 *  are skipped by the reentrancy guard below, counted in neither total. A
 *  fourth gap, check() declining a shape it cannot state as an equality, is
 *  the one the declined counter makes visible. */
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

/** As above, where the rewritten node exists only as `this`. The clone is
 *  inside the guard: it is the whole cost of the call, and a normal build must
 *  not pay it. */
inline void verify_node_rewrite(const expr2t &before, const expr2tc &after)
{
#ifdef ENABLE_SIMPLIFIER_EQUIVALENCE_CHECK
  detail::run(before.clone(), after);
#else
  (void)before;
  (void)after;
#endif
}

/** As above, for a caller that holds the rewritten node in a container only
 *  sometimes -- a nil @p before means clone @p node instead. Choosing between
 *  them here rather than at the call site keeps the branch out of simplify(),
 *  which the complexity gate gives no room to grow. */
inline void verify_rewrite_or_node(
  const expr2tc &before,
  const expr2t &node,
  const expr2tc &after)
{
#ifdef ENABLE_SIMPLIFIER_EQUIVALENCE_CHECK
  detail::run(is_nil_expr(before) ? node.clone() : before, after);
#else
  (void)before;
  (void)node;
  (void)after;
#endif
}
} // namespace simplification_check

#endif
