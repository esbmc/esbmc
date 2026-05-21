#pragma once

#include <util/algorithms.h>
#include <util/message.h>
#include <irep2/irep2.h>

/// Detect reads of uninitialised automatic-storage locals (CWE-457).
///
/// For every tracked local `x`, the pass introduces a fresh shadow
/// `__ESBMC_defined$<function>$<n>` that mirrors `x`'s initialisation
/// state — a `bool` for scalars, or a parallel `bool[N]` for fixed-size
/// arrays of scalars. The shadow starts `false` (all elements `false` for
/// arrays), is set to `true` on every direct write (`shadow[i] = true` on
/// `x[i] = …`), is asserted `true` before every direct read, and is
/// conservatively flipped entirely to `true` when `&x` (or `&x[i]`) is
/// taken — we cannot soundly track writes through pointer aliases without
/// a full alias analysis.
///
/// Tracked variables are automatic-storage, lvalue, non-extern,
/// non-`return_value$*`, non-`__ESBMC_` locals whose type is either a
/// scalar (`_Bool`, signed/unsigned integer, fixed-point, floating-point)
/// or a fixed-size 1-D array of such a scalar whose element count is a
/// constant in `[1, kMaxShadowedArraySize]` (see the .cpp). Pointers
/// (covered under CWE-908), VLAs, multi-dim arrays, structs, unions, and
/// oversize arrays remain out of scope.
///
/// Cost: every tracked array of length `N` allocates a parallel
/// `bool[N]` shadow and emits per-element `ASSERT`/`ASSIGN` instructions
/// at reads/writes, plus a `constant_array_of(true, N)` on any address-of
/// or whole-array assignment. The `kMaxShadowedArraySize` cap exists to
/// keep this from inflating the SMT encoding on programs with very large
/// fixed-size buffers; arrays above the cap are simply not tracked
/// (matching the pre-extension behaviour for that variable).
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

  bool runOnFunction(std::pair<const irep_idt, goto_functiont> &F) override;
};
