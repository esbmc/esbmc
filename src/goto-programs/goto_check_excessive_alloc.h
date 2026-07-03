#pragma once

#include <util/algorithms.h>
#include <util/mp_arith.h>
#include <irep2/irep2.h>

/// Detect memory allocations whose requested size can exceed a configurable
/// byte bound K (CWE-789, "Memory Allocation with Excessive Size Value").
///
/// For every `ASSIGN lhs = <alloc>` whose right-hand side is a dynamic
/// allocation side-effect — `malloc`/`calloc` (calloc lowers to malloc in the
/// operational model), `realloc`, or `operator new[]` — the pass inserts
/// `ASSERT(byte_size <= K)` immediately before the assignment, with comment
/// `excessive allocation size: <fn>` and property `excessive-allocation`. If
/// symex finds a model with `byte_size > K`, that is a CWE-789 witness.
///
/// `byte_size` is the total request in bytes: `malloc`/`realloc` already carry
/// a byte size; `operator new[n]` carries an element count, so the pass scales
/// it by `sizeof(element)`.
///
/// The bound K is a policy choice, not a soundness property: the pass proves
/// "no path reaches an allocation with size > K", not "no path can exhaust
/// memory" (the latter is undecidable in general). The check is independent of
/// `--force-malloc-success`: the assertion precedes the allocation, so an
/// excessive size is reported whether or not the allocation is forced to
/// succeed.
///
/// Runs as a preprocessing algorithm, mirroring `goto_check_unchecked_return`.
class goto_check_excessive_alloc : public goto_functions_algorithm
{
public:
  goto_check_excessive_alloc(contextt &context, const BigInt &max_alloc_bytes)
    : goto_functions_algorithm(true),
      context(context),
      max_alloc_bytes(max_alloc_bytes)
  {
  }

protected:
  contextt &context;
  BigInt max_alloc_bytes;

  bool runOnFunction(std::pair<const irep_idt, goto_functiont> &F) override;
};
