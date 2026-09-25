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
/// No-op if @p expr is not an add/sub/neg root, or if its type is not one of
/// is_bv_type, is_bool_type, or is_pointer_type (floating-point excluded:
/// IEEE add is not associative). Pointer-typed chains are accepted; the
/// `chain_compatible` rule constrains descent so the resulting term list
/// stays well-typed (one pointer base, integer offsets).
///
/// @returns true if any rewrite was performed.
bool reassociate_arith(expr2tc &expr);

/// Reassociate the mul chain rooted at @p expr, in place.
///
/// Flattens nested `mul` nodes into a list of leaves, folds all constant
/// integer leaves into a single product (with bit-width truncation), and
/// rebuilds as a left-leaning mul chain. Mul has no native inverse op in
/// the IR (no `recip`), so unlike `reassociate_arith` there is no sign
/// tracking and no X/-X cancellation pass — the only transform is constant
/// product folding, which crosses chain boundaries that the per-op peephole
/// can't see (e.g. `(2*x) * (3*y) -> 6*x*y`).
///
/// **Does not descend into operands.** Same contract as `reassociate_arith`.
///
/// No-op if @p expr is not a mul root, or if its type is not is_bv_type or
/// is_bool_type (floating-point and fixedbv excluded: IEEE/fixed mul is not
/// associative; pointer-typed mul is not a valid C operation).
///
/// @returns true if any rewrite was performed.
bool reassociate_mul(expr2tc &expr);

/// Reassociate the bitand/bitor/bitxor chain rooted at @p expr, in place.
///
/// All three flatten nested same-op nodes into a leaf list and fold all
/// `constant_int2t` leaves into a single value via the corresponding
/// bitwise op. Each has its own identity/absorber pair:
///
///   bitand: identity = -1, absorber = 0
///   bitor:  identity = 0,  absorber = -1
///   bitxor: identity = 0,  no absorber
///
/// `reassociate_bitxor` additionally cancels matching `x ^ x` pairs, the
/// closest analog to add's `x + (-x) = 0` cancellation. The other two have
/// no such pairwise cancellation rule.
///
/// **Does not descend into operands.** Same contract as `reassociate_arith`.
///
/// No-op if the type is not is_bv_type or is_bool_type. Constant folding
/// is bounded to ≤64-bit constants (mirrors `do_bit_munge_operation`);
/// wider constants don't fold and the rewrite is skipped.
///
/// @returns true if any rewrite was performed.
bool reassociate_bitand(expr2tc &expr);
bool reassociate_bitor(expr2tc &expr);
bool reassociate_bitxor(expr2tc &expr);

/// Run peephole simplification on @p expr, suppressing reassociation.
///
/// Called by expr2t::simplify after each successful chain-root reassoc
/// (arith, mul, and the bitwise variants) to collapse the freshly-rebuilt
/// shapes — e.g. `add(x, neg(y))` -> `sub(x, y)` via add2t/sub2t peepholes
/// — without re-entering the chain-root reassoc path. Suppression is
/// propagated through the whole subtree via the `suppress_reassoc` flag
/// threaded through expr2t::simplify.
void simplify_no_reassoc(expr2tc &expr);

#endif
