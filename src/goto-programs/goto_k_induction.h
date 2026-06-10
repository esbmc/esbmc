#ifndef GOTO_PROGRAMS_GOTO_K_INDUCTION_H_
#define GOTO_PROGRAMS_GOTO_K_INDUCTION_H_

#include <goto-programs/goto_functions.h>

/// Per-loop k-induction transformation: havoc each loop's modified
/// variables and inject an ASSUME of the loop entry condition right
/// before the loop head. Applied to every function with a body
/// (including __ESBMC_main, which is loop-free and therefore
/// untouched, and library helpers).
///
/// Returns true when at least one loop writes an array element through a
/// pointer, which the inductive step cannot soundly havoc; the caller must
/// then disable the inductive step (see issue #5224).
bool goto_k_induction(goto_functionst &goto_functions);

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_H_ */
