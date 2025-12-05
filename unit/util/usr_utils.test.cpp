/*******************************************************************\
Module: Unit tests for usr_utils.h

Notes:
    Tests for user-friendly function name to Clang USR format conversion
    and vice versa. Tests cover basic functions, namespaced functions,
    class methods, file-scoped functions, and composite cases.
\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <util/usr_utils.h>
#include <string>

TEST_CASE(
  "user_name_to_usr converts simple function names correctly",
  "[util][usr_utils]")
{
  REQUIRE(user_name_to_usr("func") == "c:@F@func#");
  REQUIRE(user_name_to_usr("main") == "c:@F@main#");
  REQUIRE(user_name_to_usr("test_function") == "c:@F@test_function#");
}

TEST_CASE(
  "user_name_to_usr converts namespaced functions correctly",
  "[util][usr_utils]")
{
  REQUIRE(user_name_to_usr("N@ns@func") == "c:@N@ns@F@func#");
  REQUIRE(
    user_name_to_usr("N@outer@N@inner@func") == "c:@N@outer@N@inner@F@func#");
}

TEST_CASE(
  "user_name_to_usr converts class methods correctly",
  "[util][usr_utils]")
{
  REQUIRE(user_name_to_usr("S@Class@method") == "c:@S@Class@F@method#");
  REQUIRE(user_name_to_usr("S@MyClass@getValue") == "c:@S@MyClass@F@getValue#");
}

TEST_CASE(
  "user_name_to_usr converts file-scoped functions correctly",
  "[util][usr_utils]")
{
  REQUIRE(user_name_to_usr("file.c@func") == "c:file.c@F@func#");
  REQUIRE(user_name_to_usr("test.cpp@helper") == "c:test.cpp@F@helper#");
  REQUIRE(user_name_to_usr("src/main.c@init") == "c:src/main.c@F@init#");
}

TEST_CASE(
  "user_name_to_usr converts composite cases correctly",
  "[util][usr_utils]")
{
  REQUIRE(user_name_to_usr("file.c@N@ns@func") == "c:file.c@N@ns@F@func#");
  REQUIRE(
    user_name_to_usr("test.cpp@S@Class@method") ==
    "c:test.cpp@S@Class@F@method#");
  REQUIRE(
    user_name_to_usr("file.c@N@ns@S@Class@method") ==
    "c:file.c@N@ns@S@Class@F@method#");
}

TEST_CASE(
  "user_name_to_usr handles already formatted USR names (passthrough)",
  "[util][usr_utils]")
{
  // Non-file-scoped USRs (c:@...)
  REQUIRE(user_name_to_usr("c:@F@func#") == "c:@F@func#");
  REQUIRE(user_name_to_usr("c:@N@ns@F@func#") == "c:@N@ns@F@func#");
  REQUIRE(user_name_to_usr("c:@S@Class@F@method#") == "c:@S@Class@F@method#");

  // Test without trailing # - should add it
  REQUIRE(user_name_to_usr("c:@F@func") == "c:@F@func#");
  REQUIRE(user_name_to_usr("c:@N@ns@F@func") == "c:@N@ns@F@func#");

  // File-scoped USRs (c:file@...) - critical for --show-loops -> --unwindsetname workflow
  REQUIRE(user_name_to_usr("c:file.c@F@func#") == "c:file.c@F@func#");
  REQUIRE(
    user_name_to_usr("c:test.cpp@N@ns@F@func#") == "c:test.cpp@N@ns@F@func#");
  REQUIRE(
    user_name_to_usr("c:file.c@S@Class@F@method#") ==
    "c:file.c@S@Class@F@method#");

  // Without trailing # - should add it
  REQUIRE(user_name_to_usr("c:file.c@F@func") == "c:file.c@F@func#");
}

TEST_CASE("user_name_to_usr handles edge cases", "[util][usr_utils]")
{
  REQUIRE(user_name_to_usr("") == "");
  // Incomplete scope markers are treated as function names
  REQUIRE(user_name_to_usr("N@") == "c:@F@N#");
  REQUIRE(user_name_to_usr("S@") == "c:@F@S#");
}

TEST_CASE(
  "usr_to_user_name converts simple function USRs correctly",
  "[util][usr_utils]")
{
  REQUIRE(usr_to_user_name("c:@F@func#") == "func");
  REQUIRE(usr_to_user_name("c:@F@main#") == "main");
  REQUIRE(usr_to_user_name("c:@F@test_function#") == "test_function");

  // Test without trailing #
  REQUIRE(usr_to_user_name("c:@F@func") == "func");
}

TEST_CASE(
  "usr_to_user_name converts namespaced function USRs correctly",
  "[util][usr_utils]")
{
  REQUIRE(usr_to_user_name("c:@N@ns@F@func#") == "N@ns@func");
  REQUIRE(usr_to_user_name("c:@N@std@F@cout#") == "N@std@cout");
  REQUIRE(
    usr_to_user_name("c:@N@outer@N@inner@F@func#") == "N@outer@N@inner@func");
}

TEST_CASE(
  "usr_to_user_name converts class method USRs correctly",
  "[util][usr_utils]")
{
  REQUIRE(usr_to_user_name("c:@S@Class@F@method#") == "S@Class@method");
  REQUIRE(usr_to_user_name("c:@S@MyClass@F@getValue#") == "S@MyClass@getValue");
}

TEST_CASE(
  "usr_to_user_name converts file-scoped function USRs correctly",
  "[util][usr_utils]")
{
  // File-scoped USRs should be converted back to user-friendly format
  REQUIRE(usr_to_user_name("c:file.c@F@func#") == "file.c@func");
  REQUIRE(usr_to_user_name("c:test.cpp@F@helper#") == "test.cpp@helper");
  REQUIRE(usr_to_user_name("c:src/main.c@F@init#") == "src/main.c@init");
}

TEST_CASE(
  "usr_to_user_name converts composite file-scoped USRs correctly",
  "[util][usr_utils]")
{
  // File-scoped composite USRs should be properly converted
  REQUIRE(usr_to_user_name("c:file.c@N@ns@F@func#") == "file.c@N@ns@func");
  REQUIRE(
    usr_to_user_name("c:test.cpp@S@Class@F@method#") ==
    "test.cpp@S@Class@method");
  REQUIRE(
    usr_to_user_name("c:file.c@N@ns@S@Class@F@method#") ==
    "file.c@N@ns@S@Class@method");
}

TEST_CASE(
  "usr_to_user_name handles non-USR input (passthrough)",
  "[util][usr_utils]")
{
  REQUIRE(usr_to_user_name("not_a_usr") == "not_a_usr");
  REQUIRE(usr_to_user_name("some_function") == "some_function");
  REQUIRE(usr_to_user_name("") == "");
}

TEST_CASE(
  "usr_to_user_name handles malformed USR input gracefully",
  "[util][usr_utils]")
{
  // USR prefix but incomplete
  REQUIRE(usr_to_user_name("c:@") == "c:@");
  REQUIRE(usr_to_user_name("c:@F@") == "c:@F@");

  // Missing c:@ prefix but has @ symbols - returned as-is
  REQUIRE(usr_to_user_name("@F@func#") == "@F@func#");
}

TEST_CASE("round-trip conversion preserves function names", "[util][usr_utils]")
{
  // All function name formats should round-trip correctly
  std::vector<std::string> test_names = {
    "func",
    "N@ns@func",
    "S@Class@method",
    "N@outer@N@inner@func",
    "N@ns@S@Class@method",
    "file.c@func",
    "test.cpp@helper",
    "file.c@N@ns@func",
    "file.c@S@Class@method",
    "file.c@N@ns@S@Class@method"};

  for (const auto &name : test_names)
  {
    std::string usr = user_name_to_usr(name);
    std::string recovered = usr_to_user_name(usr);
    REQUIRE(recovered == name);
  }
}

TEST_CASE(
  "functions handle string_view parameters correctly",
  "[util][usr_utils]")
{
  // Test with string literals (convertible to string_view)
  REQUIRE(user_name_to_usr("func") == "c:@F@func#");
  REQUIRE(usr_to_user_name("c:@F@func#") == "func");

  // Test with std::string (convertible to string_view)
  std::string user_name = "N@ns@func";
  std::string usr_name = "c:@N@ns@F@func#";
  REQUIRE(user_name_to_usr(user_name) == usr_name);
  REQUIRE(usr_to_user_name(usr_name) == user_name);

  // Test with explicit string_view
  std::string_view user_view = "S@Class@method";
  std::string_view usr_view = "c:@S@Class@F@method#";
  REQUIRE(user_name_to_usr(user_view) == "c:@S@Class@F@method#");
  REQUIRE(usr_to_user_name(usr_view) == "S@Class@method");

  // Test with substring views
  std::string long_name = "prefix_N@ns@func_suffix";
  std::string_view name_view(long_name.data() + 7, 9); // "N@ns@func"
  REQUIRE(user_name_to_usr(name_view) == "c:@N@ns@F@func#");
}

TEST_CASE("edge case: multiple consecutive scope markers", "[util][usr_utils]")
{
  // Nested namespaces
  REQUIRE(user_name_to_usr("N@a@N@b@N@c@func") == "c:@N@a@N@b@N@c@F@func#");
  REQUIRE(usr_to_user_name("c:@N@a@N@b@N@c@F@func#") == "N@a@N@b@N@c@func");

  // Class in namespace
  REQUIRE(
    user_name_to_usr("N@ns@S@Class@method") == "c:@N@ns@S@Class@F@method#");
  REQUIRE(
    usr_to_user_name("c:@N@ns@S@Class@F@method#") == "N@ns@S@Class@method");
}

TEST_CASE("edge case: special characters in names", "[util][usr_utils]")
{
  // Underscores and numbers
  REQUIRE(user_name_to_usr("test_func_123") == "c:@F@test_func_123#");
  REQUIRE(
    user_name_to_usr("S@MyClass_v2@method_1") == "c:@S@MyClass_v2@F@method_1#");

  // Round-trip
  std::string usr = user_name_to_usr("N@ns_v2@S@Class_A@method_B");
  REQUIRE(usr_to_user_name(usr) == "N@ns_v2@S@Class_A@method_B");
}
