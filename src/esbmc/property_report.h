#ifndef CPROVER_ESBMC_PROPERTY_REPORT_H
#define CPROVER_ESBMC_PROPERTY_REPORT_H

#include <goto-programs/property_verdict.h>

#include <map>
#include <set>
#include <string>
#include <vector>

/// One line of the property report: a verdict together with everything needed
/// to order and print it.
struct property_rowt
{
  std::string id;
  std::string file;
  std::string function;
  std::string description;
  std::string note;
  unsigned line = 0;
  unsigned column = 0;
  bool library = false;
  property_verdictt verdict = property_verdictt::NotChecked;
};

/// How many properties reached each verdict, and how wide the id and line
/// columns must be to line the rows up.
struct property_countst
{
  size_t passed = 0;
  size_t failed = 0;
  size_t unknown = 0;
  size_t not_checked = 0;
  size_t id_width = 0;
  size_t line_width = 0;

  /// Whether any property reached a verdict at all.
  bool anything_decided() const
  {
    return passed > 0 || failed > 0 || unknown > 0;
  }
};

property_countst count_properties(const std::vector<property_rowt> &rows);

/// Fixed-width label for a verdict, e.g. "PASSED", "NOT CHECKED".
const char *verdict_label(property_verdictt);

class goto_functionst;

/// Source files whose assertions come from a hidden function body, i.e. from
/// one of ESBMC's own operational models rather than the user's code.
std::set<std::string>
collect_library_assertion_files(const goto_functionst &goto_functions);

/// Orders \p verdicts by source position -- the table itself is keyed on the
/// property description, which sorts alphabetically -- and gives each row a
/// stable id of the form <function>.<class>.<n>, numbered per function and
/// property class in source order.
///
/// Files in \p library_files sort last, so ESBMC's own operational models do
/// not push the user's code to the bottom of the report; their absolute paths
/// would otherwise sort ahead of a relative one.
///
/// The id is a handle for reading and diffing reports. It is deliberately not
/// the number --claim takes: that one counts GOTO assert instructions in
/// function_map order (goto-programs/set_claims.cpp), which is a different
/// sequence from the source order used here.
std::vector<property_rowt> build_property_rows(
  const std::map<std::string, property_resultt> &verdicts,
  const std::set<std::string> &library_files);

#endif
