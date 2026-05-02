#ifndef UTIL_EXPR_REASSOCIATE_H_
#define UTIL_EXPR_REASSOCIATE_H_

#include <irep2/irep2.h>

/// Reassociate add/sub/neg chains in @p expr, in place.
///
/// Linearizes each add/sub/neg subtree into a list of signed terms, folds
/// constants (with bit-width truncation), cancels matching X / -X pairs,
/// and rebuilds the result as a pure add chain — letting expr2t::simplify()
/// collapse `add(x, neg(y))` back to `sub(x, y)` via its existing peepholes.
///
/// Called from both `goto_reassociate` (whole-program canonicalization
/// before symex) and `ssa_reassociate` (post-symex, post-substitution
/// cleanup of the SSA equation).
///
/// Skips overflow_* and address_of nodes to preserve their semantics.
/// Restricted to is_bv_type / is_bool_type — float reassoc is unsound
/// (IEEE non-associativity).
///
/// @returns true if any rewrite was performed.
bool reassociate_arith(expr2tc &expr);

#endif
