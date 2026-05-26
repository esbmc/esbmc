#pragma once

#include <util/algorithms.h>
#include <irep2/irep2.h>

/// Detect uses of a fallible call's return value that are not first
/// constrained to the call's success domain (CWE-252, "Unchecked Return
/// Value").
///
/// For every `FUNCTION_CALL r = f(...)` whose callee `f` is on the fixed
/// whitelist in `<util/fallible_calls.h>`, the pass scans forward over the
/// containing function's goto-program tracking `r` (and any propagation
/// chain `r' = (cast)? r`). The chain ends when the tracked variable
/// appears in:
///
///   - a GOTO guard or `__ESBMC_assume` guard — the use is path-narrowing,
///     leave it alone;
///   - any other read position (RHS of ASSIGN, function-call argument,
///     dereference target, RETURN value, ASSERT guard, OTHER such as
///     FREE) — insert an `ASSERT(success_pred(tracked))` immediately
///     before that instruction, with comment
///     `unchecked return value of <fn>: <r>` and property
///     `unchecked-return-value`;
///   - a re-assignment to a non-propagation right-hand side — the
///     binding is dead, stop silently.
///
/// v1 limitations (documented for follow-up issues):
///   - intra-procedural only;
///   - only direct-symbol callees and direct-symbol return values are
///     considered; calls that discard the return (`pthread_mutex_lock(&m);`
///     with no LHS) are skipped — flagging them needs a synthesised
///     binding;
///   - the propagation chain is a single alias hop (`tmp` → user-visible
///     binding); after the first user-visible assignment, the synthetic
///     temporary is no longer tracked, so a second indirection
///     (`q = p; *p = 1;`) misses the use of `p`;
///   - `malloc` and `realloc` are lowered to side-effect expressions
///     by the Clang frontend and are not reached by this pass —
///     unchecked uses of their results are already caught as CWE-476;
///   - operational models that hard-code a return value (e.g.
///     `pthread_mutex_lock_noassert` always returns 0) make the inserted
///     assertion vacuously true; the check is meaningful only for callees
///     whose OM leaves the return nondet.
///
/// Runs as a preprocessing algorithm, mirroring `goto_check_uninit_vars`.
class goto_check_unchecked_return : public goto_functions_algorithm
{
public:
  explicit goto_check_unchecked_return(contextt &context)
    : goto_functions_algorithm(true), context(context)
  {
  }

protected:
  contextt &context;

  bool runOnFunction(std::pair<const irep_idt, goto_functiont> &F) override;
};
