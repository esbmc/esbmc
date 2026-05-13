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

// Returns the list of CWE numeric ids associated with `comment`.
// `comment` is the freeform assertion text (e.g. "dereference failure:
// NULL pointer"). Matching is first-match-wins over an internal table whose
// keys are ordered longest-substring-first. Returns an empty vector when no
// rule matches.
std::vector<unsigned> cwe_for(const std::string &comment);

// Formats a list of ids as "CWE-476, CWE-125". Returns the empty string when
// `ids` is empty.
std::string format_cwe_list(const std::vector<unsigned> &ids);

// Returns the short MITRE name for `id` (e.g. 476 -> "NULL Pointer
// Dereference"). Returns an empty view for unknown ids.
std::string_view cwe_name(unsigned id);

#endif
