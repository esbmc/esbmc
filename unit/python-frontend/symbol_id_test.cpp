#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <python-frontend/symbol_id.h>

TEST_CASE("symbol_id::from_string parses valid input", "[symbol_id]")
{
  SECTION("Full format: filename, class and function")
  {
    auto id = symbol_id::from_string("py:main.py@C@Animal@F@speak");

    REQUIRE(id.get_prefix() == "py:");
    REQUIRE(id.get_filename() == "main.py");
    REQUIRE(id.get_class() == "Animal");
    REQUIRE(id.get_function() == "speak");
  }

  SECTION("Only filename")
  {
    auto id = symbol_id::from_string("py:script.py");

    REQUIRE(id.get_prefix() == "py:");
    REQUIRE(id.get_filename() == "script.py");
    REQUIRE(id.get_class().empty());
    REQUIRE(id.get_function().empty());
  }

  SECTION("Filename and class only")
  {
    auto id = symbol_id::from_string("py:module.py@C@MyClass");

    REQUIRE(id.get_prefix() == "py:");
    REQUIRE(id.get_filename() == "module.py");
    REQUIRE(id.get_class() == "MyClass");
    REQUIRE(id.get_function().empty());
  }

  SECTION("Filename and function only")
  {
    auto id = symbol_id::from_string("py:utils.py@F@helper");

    REQUIRE(id.get_prefix() == "py:");
    REQUIRE(id.get_filename() == "utils.py");
    REQUIRE(id.get_class().empty());
    REQUIRE(id.get_function() == "helper");
  }

  SECTION("Invalid prefix")
  {
    auto id = symbol_id::from_string("cpp:main.cpp@C@A@F@f");

    // Should return empty symbol_id (default-constructed)
    REQUIRE(id.get_prefix() == "py:"); // Default prefix
    REQUIRE(id.get_filename().empty());
    REQUIRE(id.get_class().empty());
    REQUIRE(id.get_function().empty());
  }

  SECTION("Malformed string (missing values)")
  {
    auto id = symbol_id::from_string("py:file.py@C@@F@");

    REQUIRE(id.get_filename() == "file.py");
    REQUIRE(id.get_class().empty());    // nothing after @C@
    REQUIRE(id.get_function().empty()); // nothing after @F@
  }
}
