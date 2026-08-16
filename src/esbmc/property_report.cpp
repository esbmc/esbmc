#include <esbmc/property_report.h>
#include <goto-programs/goto_functions.h>
#include <util/base/cwe_mapping.h>

#include <algorithm>
#include <tuple>
#include <unordered_map>

namespace
{
/// The property class an id is built from. cwe_rule_for() is the single source
/// of truth already shared with the SARIF, JSON and GraphML emitters; its
/// fallback id names the tool rather than the check, which reads poorly in a
/// per-property id, so a plain assertion is called what the user wrote.
std::string property_class(const std::string &description)
{
  const std::string id = cwe_rule_for(description).sarif_id;
  return id == "esbmc-assertion" ? "assertion" : id;
}
} // namespace

/// The last enumerator falls out to the return below rather than returning in
/// its own arm, so the function needs no unreachable trailing statement to
/// satisfy -Wreturn-type.
const char *verdict_label(property_verdictt verdict)
{
  switch (verdict)
  {
  case property_verdictt::NotChecked:
    return "NOT CHECKED";
  case property_verdictt::Passed:
    return "PASSED";
  case property_verdictt::Unknown:
    return "UNKNOWN";
  case property_verdictt::Failed:
    break;
  }
  return "FAILED";
}

property_countst count_properties(const std::vector<property_rowt> &rows)
{
  property_countst counts;
  for (const auto &row : rows)
  {
    switch (row.verdict)
    {
    case property_verdictt::NotChecked:
      ++counts.not_checked;
      break;
    case property_verdictt::Passed:
      ++counts.passed;
      break;
    case property_verdictt::Unknown:
      ++counts.unknown;
      break;
    case property_verdictt::Failed:
      ++counts.failed;
      break;
    }
    counts.id_width = std::max(counts.id_width, row.id.size() + 2); // brackets
    counts.line_width =
      std::max(counts.line_width, std::to_string(row.line).size());
  }
  return counts;
}

std::set<std::string>
collect_library_assertion_files(const goto_functionst &goto_functions)
{
  std::set<std::string> files;
  for (const auto &fn : goto_functions.function_map)
    if (fn.second.body.hide)
      for (const auto &instruction : fn.second.body.instructions)
        if (instruction.is_assert())
          files.insert(instruction.location.get_file().as_string());

  return files;
}

std::vector<property_rowt> build_property_rows(
  const std::map<std::string, property_resultt> &verdicts,
  const std::set<std::string> &library_files)
{
  std::vector<property_rowt> rows;
  rows.reserve(verdicts.size());

  for (const auto &[key, result] : verdicts)
  {
    property_rowt row;
    row.library = library_files.count(result.loc.file) != 0;
    row.file = result.loc.file;
    row.function = result.loc.function;
    row.line = result.loc.line;
    row.column = result.loc.column;
    row.note = result.note;
    row.verdict = result.verdict;
    // A property recorded before this ran carries no structured location; the
    // key still describes it, so fall back to that rather than dropping a row.
    row.description =
      result.loc.description.empty() ? key : result.loc.description;
    // Some checks are emitted with surrounding whitespace, which would break
    // the column the padding exists to line up.
    const size_t first = row.description.find_first_not_of(" \t\n");
    if (first == std::string::npos)
      row.description.clear();
    else
      row.description = row.description.substr(
        first, row.description.find_last_not_of(" \t\n") - first + 1);
    rows.push_back(std::move(row));
  }

  std::sort(
    rows.begin(),
    rows.end(),
    [](const property_rowt &a, const property_rowt &b) {
      return std::tie(
               a.library, a.file, a.function, a.line, a.column, a.description) <
             std::tie(
               b.library, b.file, b.function, b.line, b.column, b.description);
    });

  std::unordered_map<std::string, unsigned> counters;
  for (auto &row : rows)
  {
    // Module-level properties belong to no function; naming them after one
    // would point the reader at code they are not in.
    const std::string stem =
      (row.function.empty() ? std::string("global") : row.function) + "." +
      property_class(row.description);
    row.id = stem + "." + std::to_string(++counters[stem]);
  }

  return rows;
}
