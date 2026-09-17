#ifndef GOTO_PROGRAMS_GOTO_K_INDUCTION_H_
#define GOTO_PROGRAMS_GOTO_K_INDUCTION_H_

#include <goto-programs/goto_functions.h>
#include <util/symtab/namespace.h>

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

/// Find the first instruction at or after \p loop_head that was NOT inserted
/// by k-induction's preamble (havoc + entry-condition assume). For loops
/// whose loop_head pointed at an IF before k-induction ran, this returns
/// the (now shifted) IF; for do-while loops it returns the first real body
/// instruction. Stops at \p loop_exit defensively — every preamble
/// instruction is inside the discovered loop, so we should never reach
/// the back-edge while skipping.
goto_programt::targett skip_inductive_preamble(
  goto_programt::targett loop_head,
  goto_programt::targett loop_exit);

/// Stamp every back edge with a loop id, starting from 1. The back edge is
/// the one instruction of a loop goto_k_induction neither copies nor moves, so
/// the id names the same loop before and after the transform.
void stamp_loop_back_edges(goto_functionst &goto_functions);

void stamp_loop(goto_programt::instructiont &instruction, unsigned loop);

/// The loop id stamped on @p instruction, or 0.
unsigned stamped_loop(const goto_programt::instructiont &instruction);

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_H_ */
