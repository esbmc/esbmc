/*******************************************************************\
Module: Unit tests for xmlt
Author: Thomas Kiley
\*******************************************************************/

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <util/xml.h>

TEST_CASE("xml_equal", "[core][util][xml]")
{
  SECTION("Empty xml")
  {
    xmlt a;
    xmlt b;
    REQUIRE(a == b);
    REQUIRE_FALSE(a != b);
  }
  SECTION("Matching node")
  {
    xmlt a{"a"};
    a.data = "hello";
    a.attributes = {{"a", "b"}, {"b", "c"}};
    xmlt b{"a"};
    b.data = "hello";
    b.attributes = {{"a", "b"}, {"b", "c"}};

    REQUIRE(a == b);
    REQUIRE_FALSE(a != b);
  }
  SECTION("non-matching node")
  {
    xmlt a{"a"};
    a.data = "hello";
    a.attributes = {{"a", "b"}, {"b", "c"}};
    SECTION("Different name")
    {
      xmlt b{"b"};
      b.data = "hello";
      b.attributes = {{"a", "b"}, {"b", "c"}};

      REQUIRE_FALSE(a == b);
      REQUIRE(a != b);
    }
    SECTION("Different data")
    {
      xmlt b{"b"};
      b.data = "world";
      b.attributes = {{"a", "b"}, {"b", "c"}};

      REQUIRE_FALSE(a == b);
      REQUIRE(a != b);
    }
    SECTION("Different attributes")
    {
      xmlt b{"b"};
      b.data = "world";
      b.attributes = {{"a", "b"}, {"b", "d"}};

      REQUIRE_FALSE(a == b);
      REQUIRE(a != b);
    }
  }
  SECTION("Matching children")
  {
    xmlt a{"a"};
    a.elements = {xmlt{"b"}};
    xmlt b{"a"};
    b.elements = {xmlt{"b"}};

    REQUIRE(a == b);
    REQUIRE_FALSE(a != b);
  }
  SECTION("Non-matching children")
  {
    xmlt a{"a"};
    a.elements = {xmlt{"b"}};
    SECTION("Different child")
    {
      xmlt b{"a"};
      a.elements = {xmlt{"c"}};

      REQUIRE_FALSE(a == b);
      REQUIRE(a != b);
    }
    SECTION("Different sub child")
    {
      xmlt b{"a"};
      xmlt sub_child{"b"};
      sub_child.elements = {xmlt{"d"}};
      a.elements = {sub_child};

      REQUIRE_FALSE(a == b);
      REQUIRE(a != b);
    }
  }
}

TEST_CASE("xml_unescape", "[core][util][xml]")
{
  SECTION("No entities")
  {
    REQUIRE(xmlt::unescape("plain text") == "plain text");
  }
  SECTION("Named entities")
  {
    REQUIRE(xmlt::unescape("a&gt;b&lt;c&amp;d") == "a>b<c&d");
  }
  SECTION("Numeric entity")
  {
    REQUIRE(xmlt::unescape("&#65;&#66;") == "AB");
  }
  SECTION("Unterminated entity at end of input")
  {
    // The trailing "&gt" has no ';': the inner scan reaches end() and the
    // loop must stop without advancing the outer iterator past end()
    // (which was undefined behaviour / an out-of-bounds read before the
    // fix). The malformed tail is dropped rather than crashing.
    REQUIRE(xmlt::unescape("ok&gt") == "ok");
  }
  SECTION("Lone ampersand at end of input")
  {
    REQUIRE(xmlt::unescape("trailing&") == "trailing");
  }
}
