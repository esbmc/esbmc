#ifndef GOTO_PROGRAMS_GOTO_LOGIC_FUNCTIONS_H_
#define GOTO_PROGRAMS_GOTO_LOGIC_FUNCTIONS_H_

#include <goto-programs/goto_functions.h>
#include <util/symtab/context.h>

/// Prefix of the uninterpreted application symex models for a call to an
/// __ESBMC_logic function.
extern const char *const kLogicAppPrefix;

/// Lower calls to __ESBMC_logic functions. A call outside a logic body becomes
/// an uninterpreted application plus one unfolding of the definition, assumed;
/// calls inside a body stay uninterpreted. Runs after contracts and loop
/// invariants, which re-issue such calls where their clauses are evaluated.
void goto_logic_functions(goto_functionst &goto_functions, contextt &context);

#endif
