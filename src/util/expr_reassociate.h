#ifndef UTIL_EXPR_REASSOCIATE_H_
#define UTIL_EXPR_REASSOCIATE_H_

#include <irep2/irep2.h>

/// Reassociate the add/sub/neg chain rooted at @p expr, in place.
///
/// Linearizes the chain into a list of signed terms, folds constants (with
/// bit-width truncation), cancels matching X / -X pairs, and rebuilds as a
/// pure add chain — letting subsequent peephole simplification collapse
/// `add(x, neg(y))` back to `sub(x, y)`.
///
/// **Does not descend into operands.** Operand canonicalization is the
/// caller's responsibility. Called by expr2t::simplify(), which walks
/// operands bottom-up before invoking this exactly once at each chain root.
///
/// No-op if @p expr is not an add/sub/neg root, or if its type is not
/// is_bv_type / is_bool_type (floating-point excluded: IEEE add is not
/// associative; pointer arithmetic excluded: mixed pointer + int operand
/// types break the homogeneous-chain invariant).
///
/// @returns true if any rewrite was performed.
bool reassociate_arith(expr2tc &expr);

/// Run peephole simplification on @p expr, suppressing reassociation.
///
/// Used internally by reassociate_arith's rebuild step to collapse the
/// freshly-built `add(x, neg(y))` shapes into `sub(x, y)` via the existing
/// add2t/sub2t peepholes, without re-entering the chain-root reassoc
/// path in expr2t::simplify.
void simplify_no_reassoc(expr2tc &expr);

#endif
