#include <util/cwe_mapping.h>

#include <algorithm>
#include <map>
#include <string>
#include <string_view>

// ESBMC property-violation -> CWE mapping, pinned to CWE 4.20 (2024-11-19).
//
// The starting point was Table 4 of arxiv:2311.05281; ids whose Vulnerability
// Mapping Usage is Discouraged or Prohibited in CWE 4.20 are dropped:
//   391 (Prohibited), 119, 788, 690, 20, 682, 755, 664 (Discouraged).
// See docs/cwe-mapping.md for the full rationale.
//
// The table is sorted longest-substring-first at load time so that, e.g.,
// "invalidated dynamic object freed" matches before "invalidated dynamic
// object" regardless of declaration order below.

namespace
{
struct entry_t
{
  // string_view (not raw `const char *`) so the length is captured at
  // construction from the underlying string literal — no runtime strlen,
  // no risk of an over-read on a non-null-terminated input (Codacy /
  // Flawfinder CWE-126).
  std::string_view substring;
  cwe_rule_t rule;
};

const std::vector<entry_t> &rules_table()
{
  static const std::vector<entry_t> table = [] {
    std::vector<entry_t> t = {
      // Pointer dereference failures.
      {"dereference failure: invalidated dynamic object freed",
       {"invalidated-dynamic-object-freed",
        "Freed dynamic object dereference",
        {415, 416, 590, 761, 825}}},
      {"dereference failure: invalid pointer freed",
       {"invalid-pointer-freed",
        "Free of invalid pointer",
        {415, 416, 590, 761, 825}}},
      {"dereference failure: invalidated dynamic object",
       {"invalidated-dynamic-object",
        "Dereference of invalidated dynamic object",
        {416, 825}}},
      {"dereference failure: accessed expired variable pointer",
       {"expired-pointer-dereference",
        "Expired pointer dereference",
        {416, 562, 825}}},
      {"dereference failure: free() of non-dynamic memory",
       {"free-non-dynamic-memory", "free() of non-dynamic memory", {590, 761}}},
      {"dereference failure: forgotten memory",
       {"memory-leak", "Forgotten memory (leak)", {401}}},
      {"dereference failure: NULL pointer",
       {"null-pointer-dereference", "NULL pointer dereference", {476}}},
      {"dereference failure: memset of memory segment",
       {"memset-out-of-bounds", "memset out of bounds", {120, 125, 787}}},
      {"dereference failure on memcpy: reading memory segment",
       {"memcpy-out-of-bounds", "memcpy out of bounds", {120, 125, 787}}},
      {"dereference failure: invalid pointer",
       {"invalid-pointer-dereference",
        "Invalid pointer dereference",
        {416, 822, 824, 908}}},
      // Free-related, not phrased as "dereference failure".
      {"Operand of free must have zero pointer offset",
       {"free-non-zero-offset",
        "free() with non-zero pointer offset",
        {590, 761}}},
      // Bounds.
      {"array bounds violated",
       {"array-bounds-violated",
        "Array bounds violated",
        {121, 125, 129, 131, 193, 787}}},
      {"Access to object out of bounds",
       {"object-out-of-bounds",
        "Access to object out of bounds",
        {125, 787, 823}}},
      // Pointer relational.
      {"Same object violation",
       {"same-object-violation",
        "Same-object pointer comparison violation",
        {469}}},
      // Arithmetic.
      {"Cast arithmetic overflow",
       {"cast-arithmetic-overflow", "Cast arithmetic overflow", {190, 191}}},
      {"arithmetic overflow",
       {"arithmetic-overflow", "Arithmetic overflow", {190, 191}}},
      {"division by zero", {"division-by-zero", "Division by zero", {369}}},
      {"NaN on", {"nan", "NaN result", {681}}},
      {"undefined behavior on shift operation",
       {"shift-undefined-behavior", "Undefined behavior on shift", {1335}}},
      // Concurrency.
      {"atomicity violation",
       {"atomicity-violation", "Atomicity violation", {362, 366}}},
      {"data race on", {"data-race", "Data race", {362, 366}}},
      {"Deadlocked state", {"deadlock", "Deadlock", {833}}},
      // Reachability.
      {"unreachable code reached",
       {"reachable-error", "Reachable error/assertion", {617}}},
    };
    // Sort by descending substring length so that strict-substring overlaps
    // (e.g. "invalidated dynamic object freed" vs "invalidated dynamic
    // object") are resolved deterministically regardless of declaration
    // order. stable_sort keeps tied-length entries in source order.
    std::stable_sort(
      t.begin(), t.end(), [](const entry_t &a, const entry_t &b) {
        return a.substring.size() > b.substring.size();
      });
    return t;
  }();
  return table;
}

const std::map<unsigned, std::string_view> &names_map()
{
  // Short MITRE names for every id that can appear in `rules_table()`. Pinned
  // to CWE 4.20. std::map keeps lookup correct regardless of declaration
  // order — adding a new entry can't silently break a binary search.
  static const std::map<unsigned, std::string_view> m = {
    {120, "Buffer Copy without Checking Size of Input"},
    {121, "Stack-based Buffer Overflow"},
    {125, "Out-of-bounds Read"},
    {129, "Improper Validation of Array Index"},
    {131, "Incorrect Calculation of Buffer Size"},
    {190, "Integer Overflow or Wraparound"},
    {191, "Integer Underflow (Wrap or Wraparound)"},
    {193, "Off-by-one Error"},
    {362,
     "Concurrent Execution using Shared Resource with Improper "
     "Synchronization"},
    {366, "Race Condition within a Thread"},
    {369, "Divide By Zero"},
    {401, "Missing Release of Memory after Effective Lifetime"},
    {415, "Double Free"},
    {416, "Use After Free"},
    {469, "Use of Pointer Subtraction to Determine Size"},
    {476, "NULL Pointer Dereference"},
    {562, "Return of Stack Variable Address"},
    {590, "Free of Memory not on the Heap"},
    {617, "Reachable Assertion"},
    {681, "Incorrect Conversion between Numeric Types"},
    {761, "Free of Pointer not at Start of Buffer"},
    {787, "Out-of-bounds Write"},
    {822, "Untrusted Pointer Dereference"},
    {823, "Use of Out-of-range Pointer Offset"},
    {824, "Access of Uninitialized Pointer"},
    {825, "Expired Pointer Dereference"},
    {833, "Deadlock"},
    {908, "Use of Uninitialized Resource"},
    {1335, "Incorrect Bitwise Shift of Integer"},
  };
  return m;
}
} // namespace

const cwe_rule_t &cwe_rule_for(std::string_view comment)
{
  static const cwe_rule_t fallback{
    "esbmc-assertion", "ESBMC assertion violation", {}};
  for (const auto &e : rules_table())
    if (comment.find(e.substring) != std::string_view::npos)
      return e.rule;
  return fallback;
}

std::vector<unsigned> cwe_for(std::string_view comment)
{
  return cwe_rule_for(comment).cwes;
}

std::string format_cwe_list(const std::vector<unsigned> &ids)
{
  std::string out;
  for (unsigned id : ids)
  {
    if (!out.empty())
      out += ", ";
    out += "CWE-";
    out += std::to_string(id);
  }
  return out;
}

std::string_view cwe_name(unsigned id)
{
  const auto &m = names_map();
  auto it = m.find(id);
  return it == m.end() ? std::string_view{} : it->second;
}
