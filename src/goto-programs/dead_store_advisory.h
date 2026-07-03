#pragma once

#include <string>

/// One dead-store advisory (CWE-563, "Assignment to Variable without Use"):
/// an assignment whose left-hand side is never read on any subsequent path.
/// Advisory only — it is surfaced as a note and never flips the verdict.
///
/// This is a plain data type so the SARIF writer and bmc can depend on it
/// without pulling in the analysis pass (see dead_store_analysis.h).
struct dead_store_advisoryt
{
  /// Source-level name of the assigned variable.
  std::string lhs_name;
  /// Freeform comment, e.g. "dead store: assignment to x never read".
  /// Routed through util/cwe_mapping so it resolves to CWE-563.
  std::string comment;
  std::string file;
  std::string function;
  unsigned line = 0;
};
