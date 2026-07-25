#ifndef CPROVER_UTIL_CWE_MAPPING_H
#define CPROVER_UTIL_CWE_MAPPING_H

#include <string>
#include <string_view>
#include <vector>

// Maps ESBMC property-violation comment strings to MITRE CWE identifiers.
//
// Authoritative source: CWE 4.20 (published 2024-11-19).
// Only entries whose Vulnerability Mapping Usage is ALLOWED or
// ALLOWED-WITH-REVIEW are emitted; Discouraged / Prohibited ids are dropped.
// See docs/cwe-mapping.md for the full table and rationale.

// Single source of truth for one violation kind: the stable identifier used
// by SARIF output, the human-readable short description used by SARIF / docs,
// and the list of associated CWE ids.
struct cwe_rule_t
{
  // SARIF `reportingDescriptor.id` value. SARIF §3.49.6 requires `id` to be a
  // stable identifier; we use lowercase ASCII with hyphens so it is also a
  // valid SARIF `simpleName` (§3.5.4), even though `name` is not emitted.
  const char *sarif_id;
  // Human-readable text used for `shortDescription` / docs / logs. May
  // contain spaces, parentheses, slashes — anything non-`simpleName`.
  const char *short_description;
  // CWE ids associated with this violation kind (CWE 4.20).
  std::vector<unsigned> cwes;
};

// Returns the rule for `comment`. `comment` is the freeform assertion text
// (e.g. "dereference failure: NULL pointer"). Matching is first-match-wins
// over an internal table whose keys are sorted longest-substring-first at
// load time, so longer keys always win regardless of declaration order in
// the source. Returns a fallback rule with empty `cwes` and
// `sarif_id = "esbmc-assertion"` when no entry matches.
//
// Takes std::string_view so that callers passing `const char *` or
// `std::string` do not construct a temporary `std::string` (and GCC's
// -Wdangling-reference cannot mistake the returned reference for one bound
// into a temporary).
const cwe_rule_t &cwe_rule_for(std::string_view comment);

// Convenience wrapper: returns `cwe_rule_for(comment).cwes`.
std::vector<unsigned> cwe_for(std::string_view comment);

// Formats a list of ids as "CWE-476, CWE-125". Returns the empty string when
// `ids` is empty.
std::string format_cwe_list(const std::vector<unsigned> &ids);

// Returns the short MITRE name for `id` (e.g. 476 -> "NULL Pointer
// Dereference"). Returns an empty view for unknown ids.
std::string_view cwe_name(unsigned id);

#endif
