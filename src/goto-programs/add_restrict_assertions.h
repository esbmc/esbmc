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
/// Only function-parameter aliasing is checked: the pointed-to element
/// footprints (`sizeof(*p)` bytes, or one byte for incomplete/void subtypes)
/// must be disjoint. The footprint is an under-approximation of the accessed
/// region, so a genuinely disjoint use is never flagged. A pair is skipped when
/// both targets are const, since a never-modified object cannot violate the
/// contract. The check does, however, over-approximate the *modification*
/// requirement of C11 6.7.3.1p4: it fires on overlap of two writable-target
/// restrict parameters regardless of whether the body performs a modifying
/// access, so a function that only reads through aliasing non-const restrict
/// pointers is reported even though that is not undefined. The pass is opt-in
/// via `--restrict-check`.
///
/// The `#restricted` qualifier survives only on the legacy `code_typet` of the
/// function symbol, so the parameter types are read from `context`, not from the
/// migrated `goto_functiont::type`.
void add_restrict_assertions(
  contextt &context,
  goto_functionst &goto_functions);

#endif
