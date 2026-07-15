#ifndef CPROVER_GOTO_SYMEX_SARIF_H
#define CPROVER_GOTO_SYMEX_SARIF_H

#include <goto-programs/dead_store_advisory.h>
#include <goto-symex/goto_trace.h>
#include <util/namespace.h>
#include <util/options.h>
#include <vector>

// Writes the violation reported in `goto_trace`, plus any dead-store advisories
// (CWE-563), to a SARIF 2.1.0 document at the path given by
// `options["sarif-output"]`. Writes to stdout when the option value is "-".
// Violations are emitted as `result.level = "error"`; advisories as
// `result.level = "note"`. Includes the matching CWE ids (per
// util/cwe_mapping.h) as both per-rule tags and per-result taxa references into
// a "CWE" taxonomy pinned to CWE 4.20.
//
// `goto_trace` may be empty (e.g. on VERIFICATION SUCCESSFUL) so that a run
// with only advisories still produces a document.
void sarif_goto_trace(
  const optionst &options,
  const namespacet &ns,
  const goto_tracet &goto_trace,
  const std::vector<dead_store_advisoryt> &advisories = {});

#endif
