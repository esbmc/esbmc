#ifndef CPROVER_GOTO_SYMEX_SARIF_H
#define CPROVER_GOTO_SYMEX_SARIF_H

#include <goto-symex/goto_trace.h>
#include <util/namespace.h>
#include <util/options.h>

#include <string>
#include <vector>

// Writes the violation reported in `goto_trace` to a SARIF 2.1.0 document at
// the path given by `options["sarif-output"]`. Writes to stdout when the
// option value is "-". Includes the matching CWE ids (per util/cwe_mapping.h)
// as both per-rule tags and per-result taxa references into a "CWE" taxonomy
// pinned to CWE 4.20.
void sarif_goto_trace(
  const optionst &options,
  const namespacet &ns,
  const goto_tracet &goto_trace);

// One provably-dead statement/branch found by --dead-code-check.
struct dead_code_finding_t
{
  std::string message; // human-readable description (e.g. the branch guard)
  std::string file;
  unsigned line = 0; // 0 means "unknown", omitted from SARIF
};

// Writes the dead-code advisory `findings` to the SARIF 2.1.0 document at
// `options["sarif-output"]` (stdout when "-"). Findings are emitted with
// `result.level = "note"` (advisory, not an error) and taxa referencing
// CWE-561 in the same "CWE" taxonomy used by sarif_goto_trace. Does nothing
// when no SARIF output path is configured or `findings` is empty.
void sarif_dead_code(
  const optionst &options,
  const std::vector<dead_code_finding_t> &findings);

#endif
