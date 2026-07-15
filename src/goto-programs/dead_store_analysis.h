#pragma once

#include <vector>

#include <goto-programs/dead_store_advisory.h>
#include <util/algorithms.h>
#include <util/context.h>

/// Intra-procedural dead-store detector (CWE-563). Runs a backward
/// live-variable analysis over each function's GOTO control-flow graph and
/// appends one advisory per plain assignment to a tracked scalar local whose
/// written value is never read afterwards. The pass is non-mutating: the GOTO
/// program is left unchanged.
///
/// Only automatic-storage, non-`extern`, non-address-taken scalar locals are
/// tracked; excluding address-taken variables keeps the analysis sound without
/// an alias analysis (a variable whose address is never taken cannot be read
/// through a pointer). Aggregates, arrays, `return_value$*` and `__ESBMC_*`
/// symbols are out of scope.
///
/// Must run BEFORE mark_decl_as_non_det, which rewrites uninitialised DECLs
/// into `DECL; ASSIGN x = nondet`; that synthetic store would otherwise be
/// reported as a spurious dead store.
class goto_check_dead_store : public goto_functions_algorithm
{
public:
  goto_check_dead_store(
    const contextt &context,
    std::vector<dead_store_advisoryt> &out)
    : goto_functions_algorithm(false), context(context), advisories(out)
  {
  }

protected:
  const contextt &context;
  std::vector<dead_store_advisoryt> &advisories;

  bool runOnFunction(std::pair<const irep_idt, goto_functiont> &F) override;
};
