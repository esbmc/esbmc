#ifndef GOTO_PROGRAMS_GOTO_K_INDUCTION_H_
#define GOTO_PROGRAMS_GOTO_K_INDUCTION_H_

#include <goto-programs/goto_functions.h>
#include <util/symtab/context.h>

/// Per-loop k-induction transformation: havoc each loop's modified
/// variables and inject an ASSUME of the loop entry condition right
/// before the loop head. Applied to every function with a body
/// (including __ESBMC_main, which is loop-free and therefore
/// untouched, and library helpers).
///
/// A loop that writes through a pointer also havocs that storage: `*p` while
/// the loop leaves `p` alone, otherwise the named objects the whole-program
/// Andersen analysis resolves the pointer to. Returns true, after logging why,
/// when the inductive step would be unsound: a write in a reachable user
/// function can be covered neither way (issues #5224, #5230), a reachable call
/// to __VERIFIER_nondet_memory (#5593), or a loop that allocates under a leak
/// check. The caller must then disable the inductive step.
/// @p continue_past_failed_assertions is set when claims past a violation are
/// still checked (--multi-property, coverage), so that a path through a failed
/// assertion still counts as entering the loop (#7900).
/// @p zero_trip_bypass lets an execution that does not enter a loop skip its
/// havoc and entry-condition ASSUME, so the inductive step still checks the
/// claims past that loop. --termination clears it: its IS UNSAT is a
/// non-termination witness only when the ASSUME pins every path in the loop.
bool goto_k_induction(
  goto_functionst &goto_functions,
  contextt &context,
  bool continue_past_failed_assertions,
  bool zero_trip_bypass = true);

/// Whether \p id names the dispatcher the Solidity frontend builds for the
/// external calls into a contract C: `sol:@C@<C>@_ESBMC_Nondet_Extcall_<C>#`.
/// The id of a function declared in Solidity source always carries `@F@`, so
/// no source function can take this one.
bool is_solidity_dispatcher(const irep_idt &id);

/// The function goto_k_induction adds for a dispatcher a dispatch may reach
/// again: it havocs what one dispatch may write. In the inductive step symex
/// runs it in place of a dispatch past the unwinding bound, and before every
/// dispatch that no dispatch to the same dispatcher is under, so that the
/// unfolded dispatch also covers the states the cut-off ones start from.
irep_idt dispatch_havoc_id(const irep_idt &dispatcher);
bool is_dispatch_havoc(const irep_idt &id);
irep_idt dispatcher_of_havoc(const irep_idt &havoc);

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_H_ */
