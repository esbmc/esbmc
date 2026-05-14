/*******************************************************************\
Module: Unit tests for util/cwe_mapping.h
\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <util/cwe_mapping.h>

TEST_CASE("cwe_for matches NULL pointer deref", "[util][cwe_mapping]")
{
  REQUIRE(
    cwe_for("dereference failure: NULL pointer") == std::vector<unsigned>{476});
}

TEST_CASE("cwe_for matches array bounds violated", "[util][cwe_mapping]")
{
  REQUIRE(
    cwe_for("array bounds violated") ==
    std::vector<unsigned>{121, 125, 129, 131, 193, 787});
}

TEST_CASE("cwe_for matches arithmetic overflow", "[util][cwe_mapping]")
{
  REQUIRE(
    cwe_for("arithmetic overflow on `int'") == std::vector<unsigned>{190, 191});
  REQUIRE(
    cwe_for("Cast arithmetic overflow on `unsigned'") ==
    std::vector<unsigned>{190, 191});
}

TEST_CASE("cwe_for matches division by zero", "[util][cwe_mapping]")
{
  REQUIRE(cwe_for("division by zero") == std::vector<unsigned>{369});
}

TEST_CASE("cwe_for prefers longer substring on overlap", "[util][cwe_mapping]")
{
  // 'invalidated dynamic object freed' must match before
  // 'invalidated dynamic object'.
  REQUIRE(
    cwe_for("dereference failure: invalidated dynamic object freed") ==
    std::vector<unsigned>{415, 416, 590, 761, 825});
  REQUIRE(
    cwe_for("dereference failure: invalidated dynamic object") ==
    std::vector<unsigned>{416, 825});

  // Same for invalid pointer / invalid pointer freed.
  REQUIRE(
    cwe_for("dereference failure: invalid pointer freed") ==
    std::vector<unsigned>{415, 416, 590, 761, 825});
  REQUIRE(
    cwe_for("dereference failure: invalid pointer") ==
    std::vector<unsigned>{416, 822, 824, 908});
}

TEST_CASE("cwe_for matches free-related violations", "[util][cwe_mapping]")
{
  REQUIRE(
    cwe_for("dereference failure: free() of non-dynamic memory") ==
    std::vector<unsigned>{590, 761});
  REQUIRE(
    cwe_for("Operand of free must have zero pointer offset") ==
    std::vector<unsigned>{590, 761});
}

TEST_CASE("cwe_for matches reachability violation", "[util][cwe_mapping]")
{
  REQUIRE(cwe_for("unreachable code reached") == std::vector<unsigned>{617});
}

TEST_CASE("cwe_for matches data race", "[util][cwe_mapping]")
{
  REQUIRE(cwe_for("data race on x") == std::vector<unsigned>{362, 366});
}

TEST_CASE("cwe_for matches deadlock", "[util][cwe_mapping]")
{
  REQUIRE(
    cwe_for("Deadlocked state in pthread_mutex_lock") ==
    std::vector<unsigned>{833});
  REQUIRE(
    cwe_for("Deadlocked state in pthread_join") == std::vector<unsigned>{833});
}

TEST_CASE("cwe_for matches uninitialised variable", "[util][cwe_mapping]")
{
  REQUIRE(
    cwe_for("use of uninitialized variable: x") == std::vector<unsigned>{457});
}

TEST_CASE("cwe_for returns empty on unknown comment", "[util][cwe_mapping]")
{
  REQUIRE(cwe_for("").empty());
  REQUIRE(cwe_for("some unrelated assertion text").empty());
  // Unwinding bound is intentionally unmapped.
  REQUIRE(cwe_for("unwinding assertion loop 0").empty());
  REQUIRE(cwe_for("recursion unwinding assertion").empty());
}

TEST_CASE("format_cwe_list formats correctly", "[util][cwe_mapping]")
{
  REQUIRE(format_cwe_list({}).empty());
  REQUIRE(format_cwe_list({476}) == "CWE-476");
  REQUIRE(format_cwe_list({125, 787}) == "CWE-125, CWE-787");
}

TEST_CASE("cwe_name resolves known ids", "[util][cwe_mapping]")
{
  REQUIRE(cwe_name(476) == "NULL Pointer Dereference");
  REQUIRE(cwe_name(190) == "Integer Overflow or Wraparound");
  REQUIRE(cwe_name(369) == "Divide By Zero");
  REQUIRE(cwe_name(617) == "Reachable Assertion");
  REQUIRE(cwe_name(833) == "Deadlock");
  REQUIRE(cwe_name(457) == "Use of Uninitialized Variable");
  // Unknown id returns empty view.
  REQUIRE(cwe_name(0).empty());
  REQUIRE(cwe_name(99999).empty());
}

TEST_CASE(
  "cwe_rule_for returns a stable SARIF id and short description",
  "[util][cwe_mapping]")
{
  REQUIRE(
    std::string(cwe_rule_for("dereference failure: NULL pointer").sarif_id) ==
    "null-pointer-dereference");
  REQUIRE(
    std::string(
      cwe_rule_for("dereference failure: NULL pointer").short_description) ==
    "NULL pointer dereference");

  // Longest-substring-first invariant must hold regardless of declaration
  // order in cwe_mapping.cpp.
  REQUIRE(
    std::string(
      cwe_rule_for("dereference failure: invalidated dynamic object freed")
        .sarif_id) == "invalidated-dynamic-object-freed");
  REQUIRE(
    std::string(cwe_rule_for("dereference failure: invalidated dynamic object")
                  .sarif_id) == "invalidated-dynamic-object");

  // Fallback for unrecognised comments.
  REQUIRE(std::string(cwe_rule_for("").sarif_id) == "esbmc-assertion");
  REQUIRE(cwe_rule_for("").cwes.empty());
}

TEST_CASE(
  "every SARIF rule id is a valid simpleName (no spaces / punctuation)",
  "[util][cwe_mapping]")
{
  // Walk every documented comment and check the rule's sarif_id contains only
  // characters allowed by SARIF §3.5.4 (simpleName): ASCII letters, digits,
  // '.', '_', plus '-' which is also widely accepted by validators.
  for (const char *comment :
       {"dereference failure: NULL pointer",
        "dereference failure: invalid pointer freed",
        "dereference failure: invalidated dynamic object freed",
        "dereference failure: invalidated dynamic object",
        "dereference failure: accessed expired variable pointer",
        "dereference failure: invalid pointer",
        "dereference failure: free() of non-dynamic memory",
        "Operand of free must have zero pointer offset",
        "dereference failure: forgotten memory",
        "array bounds violated",
        "Access to object out of bounds",
        "dereference failure: memset of memory segment of size 4",
        "dereference failure on memcpy: reading memory segment of size 4",
        "Same object violation",
        "Cast arithmetic overflow",
        "arithmetic overflow",
        "division by zero",
        "NaN on x",
        "undefined behavior on shift operation",
        "atomicity violation",
        "data race on x",
        "Deadlocked state in pthread_mutex_lock",
        "use of uninitialized variable: foo",
        "unreachable code reached",
        ""})
  {
    const cwe_rule_t &rule = cwe_rule_for(comment);
    std::string id(rule.sarif_id);
    REQUIRE_FALSE(id.empty());
    for (char c : id)
    {
      bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
      INFO("rule id: " << id << "  offending char: " << c);
      REQUIRE(ok);
    }
  }
}

TEST_CASE(
  "every id used in cwe_for has a name in cwe_name",
  "[util][cwe_mapping]")
{
  // Sample a few representative comments and check every returned id has a
  // non-empty name — guards against drift between the rules table and the
  // names table.
  for (const char *comment :
       {"dereference failure: NULL pointer",
        "dereference failure: invalid pointer freed",
        "array bounds violated",
        "arithmetic overflow",
        "division by zero",
        "atomicity violation",
        "data race on x",
        "Deadlocked state in pthread_mutex_lock",
        "use of uninitialized variable: foo",
        "Access to object out of bounds",
        "dereference failure: memset of memory segment of size 4",
        "undefined behavior on shift operation"})
  {
    for (unsigned id : cwe_for(comment))
      REQUIRE_FALSE(cwe_name(id).empty());
  }
}
