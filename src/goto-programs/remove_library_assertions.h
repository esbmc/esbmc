#ifndef CPROVER_GOTO_PROGRAMS_REMOVE_LIBRARY_ASSERTIONS_H
#define CPROVER_GOTO_PROGRAMS_REMOVE_LIBRARY_ASSERTIONS_H

#include <goto-programs/goto_functions.h>

/// Drop every assertion stated by ESBMC's own operational models, i.e. by a
/// function whose body carries the __ESBMC_HIDE label. Must run before
/// inlining: goto_inlinet rewrites the locations of a hidden function's
/// instructions to the call site, after which model assertions are
/// indistinguishable from the user's own. For the same reason a goto binary
/// written with --full-inlining no longer has anything for this to remove.
void remove_library_assertions(goto_functionst &goto_functions);

#endif
