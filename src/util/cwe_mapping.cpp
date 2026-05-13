#include <util/cwe_mapping.h>

#include <algorithm>
#include <string>

// ESBMC property-violation -> CWE mapping, pinned to CWE 4.20 (2024-11-19).
//
// The starting point was Table 4 of arxiv:2311.05281; ids whose Vulnerability
// Mapping Usage is Discouraged or Prohibited in CWE 4.20 are dropped:
//   391 (Prohibited), 119, 788, 690, 20, 682, 755, 664 (Discouraged).
// See docs/cwe-mapping.md for the full rationale.
//
// Keys are ordered longest-substring-first so that, e.g., "invalidated
// dynamic object freed" matches before "invalidated dynamic object".

namespace
{
struct rule_t
{
  const char *substring;
  std::vector<unsigned> ids;
};

const std::vector<rule_t> &rules()
{
  static const std::vector<rule_t> table = {
    // Pointer dereference failures (longest substrings first).
    {"dereference failure: invalidated dynamic object freed",
     {415, 416, 590, 761, 825}},
    {"dereference failure: invalid pointer freed", {415, 416, 590, 761, 825}},
    {"dereference failure: invalidated dynamic object", {416, 825}},
    {"dereference failure: accessed expired variable pointer", {416, 562, 825}},
    {"dereference failure: free() of non-dynamic memory", {590, 761}},
    {"dereference failure: forgotten memory", {401}},
    {"dereference failure: NULL pointer", {476}},
    {"dereference failure: memset of memory segment", {120, 125, 787}},
    {"dereference failure on memcpy: reading memory segment", {120, 125, 787}},
    {"dereference failure: invalid pointer", {416, 822, 824, 908}},
    // Free-related, not phrased as "dereference failure".
    {"Operand of free must have zero pointer offset", {590, 761}},
    // Bounds.
    {"array bounds violated", {121, 125, 129, 131, 193, 787}},
    {"Access to object out of bounds", {125, 787, 823}},
    // Pointer relational.
    {"Same object violation", {469}},
    // Arithmetic.
    {"Cast arithmetic overflow", {190, 191}},
    {"arithmetic overflow", {190, 191}},
    {"division by zero", {369}},
    {"NaN on", {681}},
    {"undefined behavior on shift operation", {1335}},
    // Concurrency.
    {"atomicity violation", {362, 366}},
    {"data race on", {362, 366}},
    // Reachability.
    {"unreachable code reached", {617}},
  };
  return table;
}

struct name_entry_t
{
  unsigned id;
  std::string_view name;
};

const std::vector<name_entry_t> &names()
{
  // Short MITRE names for every id that can appear in `rules()`. Pinned to
  // CWE 4.20.
  static const std::vector<name_entry_t> table = {
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
    {908, "Use of Uninitialized Resource"},
    {1335, "Incorrect Bitwise Shift of Integer"},
  };
  return table;
}
} // namespace

std::vector<unsigned> cwe_for(const std::string &comment)
{
  for (const auto &rule : rules())
    if (comment.find(rule.substring) != std::string::npos)
      return rule.ids;
  return {};
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
  const auto &table = names();
  auto it = std::lower_bound(
    table.begin(),
    table.end(),
    id,
    [](const name_entry_t &e, unsigned v) { return e.id < v; });
  if (it != table.end() && it->id == id)
    return it->name;
  return {};
}
