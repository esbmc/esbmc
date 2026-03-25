#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <python-frontend/function_call_cache.h>

TEST_CASE("function_call_cache API correctness", "[python-frontend][cache]")
{
  function_call_cache cache;

  SECTION("store and retrieve possible class types")
  {
    std::vector<std::string> types = {"Dog", "Cat"};
    cache.set_possible_class_types("sym::obj", types);

    const auto *result = cache.get_possible_class_types("sym::obj");
    REQUIRE(result != nullptr);
    REQUIRE(*result == types);
  }

  SECTION("miss on unknown key returns nullptr")
  {
    REQUIRE(cache.get_possible_class_types("no_such_key") == nullptr);
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
    cache.set_math_dispatch_classification("math::sin", true);

    cache.clear();

    REQUIRE(cache.get_possible_class_types("k1") == nullptr);
    REQUIRE_FALSE(cache.get_method_exists("A::foo").has_value());
    REQUIRE_FALSE(
      cache.get_math_dispatch_classification("math::sin").has_value());
  }

  SECTION("overwrite existing entries")
  {
    cache.set_possible_class_types("key", {"Old"});
    cache.set_possible_class_types("key", {"New"});

    const auto *result = cache.get_possible_class_types("key");
    REQUIRE(result != nullptr);
    REQUIRE(*result == std::vector<std::string>{"New"});

    cache.set_method_exists("C::m", false);
    cache.set_method_exists("C::m", true);

    auto val = cache.get_method_exists("C::m");
    REQUIRE(val.has_value());
    REQUIRE(val.value() == true);
  }

  SECTION("store and retrieve math dispatch classification")
  {
    cache.set_math_dispatch_classification("math::sin", true);
    cache.set_math_dispatch_classification("other::sin", false);

    auto math_hit = cache.get_math_dispatch_classification("math::sin");
    REQUIRE(math_hit.has_value());
    REQUIRE(math_hit.value() == true);

    auto non_math_hit = cache.get_math_dispatch_classification("other::sin");
    REQUIRE(non_math_hit.has_value());
    REQUIRE(non_math_hit.value() == false);

    auto miss = cache.get_math_dispatch_classification("math::unknown");
    REQUIRE_FALSE(miss.has_value());
  }
}
