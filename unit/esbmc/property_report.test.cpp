/*******************************************************************
 Module: Property report unit tests
 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <esbmc/property_report.h>
#include <goto-programs/goto_functions.h>
#include <irep2/irep2_utils.h>

namespace
{
property_resultt result(
  property_verdictt verdict,
  std::string file,
  std::string function,
  unsigned line,
  std::string description,
  unsigned column = 0)
{
  property_resultt r;
  r.verdict = verdict;
  r.loc.file = std::move(file);
  r.loc.function = std::move(function);
  r.loc.description = std::move(description);
  r.loc.line = line;
  r.loc.column = column;
  return r;
}

std::vector<std::string> ids(const std::vector<property_rowt> &rows)
{
  std::vector<std::string> out;
  for (const auto &row : rows)
    out.push_back(row.id);
  return out;
}

/// A function whose body holds one assert at \p file, hidden or not.
goto_functiont asserting_function(const std::string &file, bool hide)
{
  goto_functiont fn;
  fn.body.hide = hide;
  auto it = fn.body.add_instruction();
  it->make_assertion(gen_true_expr());
  it->location.set_file(file);
  return fn;
}
} // namespace

TEST_CASE("verdict_label names every verdict", "[property-report]")
{
  CHECK(
    std::string(verdict_label(property_verdictt::NotChecked)) == "NOT CHECKED");
  CHECK(std::string(verdict_label(property_verdictt::Passed)) == "PASSED");
  CHECK(std::string(verdict_label(property_verdictt::Unknown)) == "UNKNOWN");
  CHECK(std::string(verdict_label(property_verdictt::Failed)) == "FAILED");
}

TEST_CASE("count_properties tallies each verdict", "[property-report]")
{
  std::vector<property_rowt> rows(5);
  rows[0].verdict = property_verdictt::Passed;
  rows[1].verdict = property_verdictt::Passed;
  rows[2].verdict = property_verdictt::Failed;
  rows[3].verdict = property_verdictt::Unknown;
  rows[4].verdict = property_verdictt::NotChecked;

  const property_countst counts = count_properties(rows);
  CHECK(counts.passed == 2);
  CHECK(counts.failed == 1);
  CHECK(counts.unknown == 1);
  CHECK(counts.not_checked == 1);
  CHECK(counts.anything_decided());
}

TEST_CASE("count_properties of nothing decides nothing", "[property-report]")
{
  const property_countst counts = count_properties({});
  CHECK(counts.passed == 0);
  CHECK(counts.id_width == 0);
  CHECK(counts.line_width == 0);
  CHECK_FALSE(counts.anything_decided());
}

TEST_CASE(
  "a table of only unchecked properties has decided nothing",
  "[property-report]")
{
  std::vector<property_rowt> rows(2);
  CHECK(count_properties(rows).not_checked == 2);
  CHECK_FALSE(count_properties(rows).anything_decided());
}

TEST_CASE("column widths span the widest row", "[property-report]")
{
  std::vector<property_rowt> rows(2);
  rows[0].id = "main.assertion.1";
  rows[0].line = 7;
  rows[1].id = "f.division-by-zero.10";
  rows[1].line = 1234;

  const property_countst counts = count_properties(rows);
  // Two wider than the id itself: the report prints it in brackets.
  CHECK(counts.id_width == rows[1].id.size() + 2);
  CHECK(counts.line_width == 4);
}

TEST_CASE("rows sort by source position", "[property-report]")
{
  const std::map<std::string, property_resultt> verdicts{
    {"c", result(property_verdictt::Passed, "a.c", "main", 20, "third")},
    {"a", result(property_verdictt::Passed, "a.c", "main", 10, "first")},
    {"b", result(property_verdictt::Passed, "a.c", "main", 10, "second", 4)},
    {"d", result(property_verdictt::Passed, "b.c", "main", 1, "fourth")}};

  const std::vector<property_rowt> rows = build_property_rows(verdicts, {});
  REQUIRE(rows.size() == 4);
  CHECK(rows[0].description == "first");
  CHECK(rows[1].description == "second");
  CHECK(rows[2].description == "third");
  CHECK(rows[3].description == "fourth");
}

TEST_CASE("library rows sort last whatever their path", "[property-report]")
{
  // An absolute path would otherwise sort ahead of the user's relative one.
  const std::map<std::string, property_resultt> verdicts{
    {"lib", result(property_verdictt::Passed, "/opt/om/stdlib.c", "f", 1, "l")},
    {"user", result(property_verdictt::Failed, "user.c", "main", 9, "u")}};

  const std::vector<property_rowt> rows =
    build_property_rows(verdicts, {"/opt/om/stdlib.c"});
  REQUIRE(rows.size() == 2);
  CHECK(rows[0].description == "u");
  CHECK_FALSE(rows[0].library);
  CHECK(rows[1].description == "l");
  CHECK(rows[1].library);
}

TEST_CASE(
  "ids are numbered per function and property class",
  "[property-report]")
{
  const std::map<std::string, property_resultt> verdicts{
    {"a", result(property_verdictt::Failed, "a.c", "main", 1, "an assertion")},
    {"b",
     result(property_verdictt::Failed, "a.c", "main", 2, "division by zero")},
    {"c", result(property_verdictt::Failed, "a.c", "main", 3, "another one")},
    {"d",
     result(property_verdictt::Failed, "a.c", "helper", 4, "yet another")}};

  // Rows sort by function before line, so helper's row leads.
  const std::vector<std::string> expected{
    "helper.assertion.1",
    "main.assertion.1",
    "main.division-by-zero.1",
    "main.assertion.2"};
  CHECK(ids(build_property_rows(verdicts, {})) == expected);
}

TEST_CASE("a property in no function is called global", "[property-report]")
{
  const std::map<std::string, property_resultt> verdicts{
    {"a", result(property_verdictt::Unknown, "a.c", "", 1, "an assertion")}};

  CHECK(build_property_rows(verdicts, {})[0].id == "global.assertion.1");
}

TEST_CASE("a row with no location falls back to its key", "[property-report]")
{
  const std::map<std::string, property_resultt> verdicts{
    {"unnamed claim", result(property_verdictt::NotChecked, "", "", 0, "")}};

  CHECK(build_property_rows(verdicts, {})[0].description == "unnamed claim");
}

TEST_CASE("surrounding whitespace is trimmed off", "[property-report]")
{
  const std::map<std::string, property_resultt> verdicts{
    {"a", result(property_verdictt::Passed, "a.c", "main", 1, "  spaced \n")},
    {"b", result(property_verdictt::Passed, "a.c", "main", 2, " \t\n")}};

  const std::vector<property_rowt> rows = build_property_rows(verdicts, {});
  CHECK(rows[0].description == "spaced");
  CHECK(rows[1].description.empty());
}

TEST_CASE("only hidden functions contribute library files", "[property-report]")
{
  goto_functionst goto_functions;
  goto_functions.function_map["om"] = asserting_function("stdlib.c", true);
  goto_functions.function_map["user"] = asserting_function("user.c", false);

  const std::set<std::string> files =
    collect_library_assertion_files(goto_functions);
  CHECK(files == std::set<std::string>{"stdlib.c"});
}

TEST_CASE(
  "a hidden function with no assert contributes nothing",
  "[property-report]")
{
  goto_functionst goto_functions;
  goto_functiont fn;
  fn.body.hide = true;
  fn.body.add_instruction()->make_skip();
  goto_functions.function_map["om"] = std::move(fn);

  CHECK(collect_library_assertion_files(goto_functions).empty());
}
