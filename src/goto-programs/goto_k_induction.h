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
/// A loop that writes through a pointer also havocs that storage: `*p` while
/// the loop leaves `p` alone, otherwise the named objects the whole-program
/// Andersen analysis resolves the pointer to. Returns true when a write in a
/// reachable user function can be covered neither way; the caller must then
/// disable the inductive step (issues #5224, #5230).
bool goto_k_induction(goto_functionst &goto_functions, const namespacet &ns);

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_H_ */
