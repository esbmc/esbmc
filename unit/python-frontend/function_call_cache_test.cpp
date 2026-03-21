#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <python-frontend/function_call_cache.h>

TEST_CASE(
  "function_call_cache API correctness",
  "[python-frontend][cache]")
{
  function_call_cache cache;

  SECTION("store and retrieve possible class types")
  {
    std::vector<std::string> types = {"Dog", "Cat"};
    cache.set_possible_class_types("sym::obj", types);

    auto result = cache.get_possible_class_types("sym::obj");
    REQUIRE(result.has_value());
    REQUIRE(result.value() == types);
  }

  SECTION("miss on unknown key returns nullopt")
  {
    auto result = cache.get_possible_class_types("no_such_key");
    REQUIRE_FALSE(result.has_value());
  }

  SECTION("store and retrieve method exists")
  {
    cache.set_method_exists("Dog::speak", true);
    cache.set_method_exists("Dog::fly", false);

    auto found = cache.get_method_exists("Dog::speak");
    REQUIRE(found.has_value());
    REQUIRE(found.value() == true);

    auto not_found = cache.get_method_exists("Dog::fly");
    REQUIRE(not_found.has_value());
    REQUIRE(not_found.value() == false);
  }

  SECTION("method exists miss returns nullopt")
  {
    auto result = cache.get_method_exists("Unknown::method");
    REQUIRE_FALSE(result.has_value());
  }

  SECTION("clear removes all entries")
  {
    cache.set_possible_class_types("k1", {"A"});
    cache.set_method_exists("A::foo", true);

    cache.clear();

    REQUIRE_FALSE(cache.get_possible_class_types("k1").has_value());
    REQUIRE_FALSE(cache.get_method_exists("A::foo").has_value());
  }

  SECTION("overwrite existing entries")
  {
    cache.set_possible_class_types("key", {"Old"});
    cache.set_possible_class_types("key", {"New"});

    auto result = cache.get_possible_class_types("key");
    REQUIRE(result.has_value());
    REQUIRE(result.value() == std::vector<std::string>{"New"});

    cache.set_method_exists("C::m", false);
    cache.set_method_exists("C::m", true);

    auto val = cache.get_method_exists("C::m");
    REQUIRE(val.has_value());
    REQUIRE(val.value() == true);
  }
}
