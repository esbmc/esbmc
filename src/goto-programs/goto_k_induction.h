#ifndef GOTO_PROGRAMS_GOTO_K_INDUCTION_H_
#define GOTO_PROGRAMS_GOTO_K_INDUCTION_H_

#include <goto-programs/goto_functions.h>
#include <util/namespace.h>

/// Per-loop k-induction transformation: havoc each loop's modified
/// variables and inject an ASSUME of the loop entry condition right
/// before the loop head. Applied to every function with a body
/// (including __ESBMC_main, which is loop-free and therefore
/// untouched, and library helpers).
///
/// For a loop that writes an array element through a pointer, Phase 2
/// (issue #5230) resolves the written pointer against the value-set
/// fixpoint built over @p ns and havocs the referenced named objects as
/// whole symbols, keeping the inductive step sound and enabled. Returns
/// true only when such a write cannot be resolved (callee parameter,
/// unknown/heap pointee); the caller must then disable the inductive step,
/// the conservative Phase 1 behaviour (issue #5224).
bool goto_k_induction(goto_functionst &goto_functions, const namespacet &ns);

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_H_ */
