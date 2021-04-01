/*******************************************************************
 Module: LRU Cache unit test
 Author: Rafael Sá Menezes
 Date: January 2019
 Test Plan:
   - Constructors
   - Base Methods
   - Integrity/Atomicity
 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <cache/containers/lru_cache.h>

// ******************** TESTS ********************
SCENARIO("lru_cache is initialized correctly", "[cache]")
{
  GIVEN("A LRU cache of size 10")
  {
    lru_cache<int> obj(10);
    REQUIRE(obj.max_size() == 10);
    REQUIRE(obj.size() == 0);
    REQUIRE_FALSE(obj.exists(0));

    WHEN("Adding elements")
    {
      obj.insert(0);

      THEN("Value should be inserted")
      {
        REQUIRE(obj.exists(0));
        REQUIRE(obj.size() == 1);
      }

      AND_THEN("Inserting same value shouldn't affect")
      {
        obj.insert(0);
        REQUIRE(obj.exists(0));
        REQUIRE(obj.size() == 1);
      }

      AND_THEN("Adding another element should add even more")
      {
        obj.insert(1);
        REQUIRE(obj.exists(0));
        REQUIRE(obj.exists(1));
        REQUIRE(obj.size() == 2);
      }
    }

    WHEN("Overflowing the cache")
    {
      for(int i = 0; i < 50; i++)
        obj.insert(i);

      THEN("Max size should be respected")
      {
        REQUIRE(obj.max_size() == 10);
        REQUIRE(obj.size() <= obj.max_size());
      }
    }
  }
}

// ** Fuzzer cases
// Test cases that were found by libfuzzer

TEST_CASE("lru fuzzing failures", "[caching]")
{
  SECTION("crash_1")
  {
    // This input was causing a memory leak because std::list is not
    // smart enough to know when it can release it's entries.
    const int input[] = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    const size_t cache_size = 200;
    lru_cache<int> obj(cache_size);
    for(const auto &i : input)
    {
      obj.insert(i);
      REQUIRE(obj.exists(i));
    }
  }
}
