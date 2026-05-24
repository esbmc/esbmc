#ifndef CPROVER_GOTO_SYMEX_SARIF_H
#define CPROVER_GOTO_SYMEX_SARIF_H

#include <goto-symex/goto_trace.h>
#include <util/namespace.h>
#include <util/options.h>

// Writes the violation reported in `goto_trace` to a SARIF 2.1.0 document at
// the path given by `options["sarif-output"]`. Writes to stdout when the
// option value is "-". Includes the matching CWE ids (per util/cwe_mapping.h)
// as both per-rule tags and per-result taxa references into a "CWE" taxonomy
// pinned to CWE 4.20.
void sarif_goto_trace(
  const optionst &options,
  const namespacet &ns,
  const goto_tracet &goto_trace);

#endif
