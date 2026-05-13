#pragma once

#include <util/algorithms.h>
#include <util/message.h>
#include <irep2/irep2.h>

/// Detect reads of uninitialised automatic-storage local scalars (CWE-457).
///
/// For every tracked scalar local `x`, the pass introduces a fresh shadow
/// boolean `__ESBMC_defined$<function>$<n>` that starts `false`, is set to
/// `true` on every direct write to `x`, is asserted `true` before every
/// direct read of `x`, and is conservatively set to `true` when `&x` is
/// taken (because we cannot soundly track writes through pointer aliases
/// without a full alias analysis).
///
/// Tracked variables are automatic-storage, lvalue, non-extern,
/// non-`return_value$*`, non-`__ESBMC_` locals whose type is a scalar
/// (`_Bool`, signed/unsigned integer, fixed-point, floating-point). Pointers,
/// arrays, structs, and unions are out of scope for v1.
///
/// The pass must run **before** `mark_decl_as_non_det`, which would
/// otherwise rewrite every uninitialised `DECL` into a `DECL; ASSIGN nondet`
/// pair, erasing the uninitialised-state distinction.
class goto_check_uninit_vars : public goto_functions_algorithm
{
public:
  explicit goto_check_uninit_vars(contextt &context)
    : goto_functions_algorithm(true), context(context)
  {
  }

protected:
  contextt &context;

  bool runOnFunction(std::pair<const dstring, goto_functiont> &F) override;
};
