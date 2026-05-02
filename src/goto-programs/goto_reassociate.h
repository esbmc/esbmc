#ifndef GOTO_PROGRAMS_GOTO_REASSOCIATE_H_
#define GOTO_PROGRAMS_GOTO_REASSOCIATE_H_

#include <goto-programs/goto_functions.h>

/// Reassociates add/sub chains in every instruction to surface constant folds
/// and X+(-X) cancellations that the local peephole simplifier in
/// expr_simplifier.cpp can't reach.
///
/// Mirrors LLVM's Reassociate pass at a coarser granularity: linearize each
/// add/sub tree into a list of signed terms, fold constants, cancel inverses,
/// rebuild a balanced tree.
///
/// Caller is responsible for skipping this when --no-simplify is set.
void goto_reassociate(goto_functionst &goto_functions);

#endif
