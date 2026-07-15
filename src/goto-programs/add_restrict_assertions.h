#ifndef CPROVER_GOTO_PROGRAMS_ADD_RESTRICT_ASSERTIONS_H
#define CPROVER_GOTO_PROGRAMS_ADD_RESTRICT_ASSERTIONS_H

#include <goto-programs/goto_functions.h>
#include <util/context.h>

/// Instrument every function that has two or more `restrict`-qualified pointer
/// parameters with an entry assertion that the objects they point at do not
/// overlap. Aliasing restrict pointers whose accessed regions overlap are
/// undefined behaviour (C11 6.7.3.1); this catches the classic case, e.g.
/// `void f(int *restrict a, int *restrict b)` called with `f(&x, &x)`.
///
/// Only function-parameter aliasing is checked: two non-null restrict pointers
/// must not have overlapping element footprints (`sizeof(*p)` bytes, or one byte
/// for incomplete/void subtypes) within a shared object. The footprint is an
/// under-approximation of the accessed region, so a genuinely disjoint use is
/// never flagged, and null parameters (which designate no object) are exempt.
///
/// The check over-approximates the *modification* clause of C11 6.7.3.1p4: it
/// fires on overlap regardless of whether the body performs a modifying access.
/// const-qualified targets are intentionally not exempt — 6.7.3.1p4 makes even a
/// const access path undefined once the shared object is modified by any means,
/// which this pass cannot rule out. The pass is opt-in via `--restrict-check`.
///
/// The `#restricted` qualifier survives only on the legacy `code_typet` of the
/// function symbol, so the parameter types are read from `context`, not from the
/// migrated `goto_functiont::type`.
void add_restrict_assertions(
  contextt &context,
  goto_functionst &goto_functions);

#endif
