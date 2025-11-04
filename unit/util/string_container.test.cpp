#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <thread>
#include <vector>
#include <set>
#include <util/string_container.h>

class StringContainerTest
{
protected:
  string_containert container;
};

// ============================================================================
// Basic Functionality Tests - char*
// ============================================================================

TEST_CASE("char* pointer: basic insertion and retrieval", "[char_ptr][basic]")
{
  string_containert container;

  unsigned id1 = container["hello"];
  unsigned id2 = container["world"];

  REQUIRE(id1 == 1); // 0 is reserved for empty string
  REQUIRE(id2 == 2);
  REQUIRE(container.c_str(id1) == std::string("hello"));
  REQUIRE(container.c_str(id2) == std::string("world"));
}

TEST_CASE("char* pointer: empty string gets index 0", "[char_ptr][basic]")
{
  string_containert container;
  unsigned empty_id = container[""];
  REQUIRE(empty_id == 0);
  REQUIRE(container.c_str(0) == std::string(""));
}

TEST_CASE(
  "char* pointer: duplicate strings return same index",
  "[char_ptr][basic]")
{
  string_containert container;

  unsigned id1 = container["test"];
  unsigned id2 = container["test"];
  unsigned id3 = container["test"];

  REQUIRE(id1 == id2);
  REQUIRE(id2 == id3);
}

TEST_CASE(
  "char* pointer: different strings get different indices",
  "[char_ptr][basic]")
{
  string_containert container;

  unsigned id1 = container["apple"];
  unsigned id2 = container["banana"];
  unsigned id3 = container["cherry"];

  REQUIRE(id1 != id2);
  REQUIRE(id2 != id3);
  REQUIRE(id1 != id3);
}

TEST_CASE("char* pointer: string stability", "[char_ptr][stability]")
{
  string_containert container;

  unsigned id = container["stable"];
  const char *ptr1 = container.c_str(id);

  // Insert more strings
  for (int i = 0; i < 100; ++i)
  {
    container["string_" + std::to_string(i)];
  }

  const char *ptr2 = container.c_str(id);
  REQUIRE(ptr1 == ptr2); // Pointer must remain the same
  REQUIRE(std::string(ptr2) == "stable");
}

// ============================================================================
// Basic Functionality Tests - std::string
// ============================================================================

TEST_CASE("std::string: basic insertion and retrieval", "[string][basic]")
{
  string_containert container;

  std::string hello_str = "hello";
  std::string world_str = "world";

  unsigned id1 = container[hello_str];
  unsigned id2 = container[world_str];

  REQUIRE(id1 == 1);
  REQUIRE(id2 == 2);
  REQUIRE(container.c_str(id1) == hello_str);
  REQUIRE(container.c_str(id2) == world_str);
}

TEST_CASE("std::string: duplicate strings return same index", "[string][basic]")
{
  string_containert container;

  std::string test_str = "test";
  unsigned id1 = container[test_str];
  unsigned id2 = container[test_str];

  REQUIRE(id1 == id2);
}

TEST_CASE("std::string: temporary string handling", "[string][basic]")
{
  string_containert container;

  unsigned id = container[std::string("temporary")];
  REQUIRE(container.c_str(id) == std::string("temporary"));
}

TEST_CASE("std::string: get_string returns reference", "[string][basic]")
{
  string_containert container;

  std::string input = "reference_test";
  unsigned id = container[input];

  const std::string &ref = container.get_string(id);
  REQUIRE(ref == input);
  REQUIRE(&ref == &container.get_string(id)); // Same reference
}

// ============================================================================
// Cross-type Compatibility Tests
// ============================================================================

TEST_CASE("mixed types: char* and std::string", "[mixed][basic]")
{
  string_containert container;

  const char *c_str = "same";
  std::string cpp_str = "same";

  unsigned id1 = container[c_str];
  unsigned id2 = container[cpp_str];

  REQUIRE(id1 == id2); // Should return same index
}

TEST_CASE("mixed types: multiple insertions", "[mixed][basic]")
{
  string_containert container;

  unsigned id1 = container["str1"];
  unsigned id2 = container[std::string("str2")];
  unsigned id3 = container["str3"];
  unsigned id4 = container[std::string("str4")];

  REQUIRE(id1 != id2);
  REQUIRE(id3 != id4);
  REQUIRE(std::string(container.c_str(id1)) == "str1");
  REQUIRE(std::string(container.c_str(id2)) == "str2");
}

// ============================================================================
// string_view Tests (Disabled until implementation)
// ============================================================================

TEST_CASE("std::string_view: basic insertion", "[string_view][basic]")
{
  string_containert container;

  std::string_view view1 = "view_test";
  std::string_view view2 = "another_view";

  unsigned id1 = container[view1];
  unsigned id2 = container[view2];

  REQUIRE(id1 != id2);
  REQUIRE(container.c_str(id1) == std::string("view_test"));
  REQUIRE(container.c_str(id2) == std::string("another_view"));
}

TEST_CASE("std::string_view: substring view", "[string_view][basic]")
{
  string_containert container;

  std::string full = "hello_world";
  std::string_view view(full.data() + 6, 5); // "world"

  unsigned id = container[view];
  REQUIRE(container.c_str(id) == std::string("world"));
}

TEST_CASE("std::string_view: duplicate strings", "[string_view][basic]")
{
  string_containert container;

  std::string source = "duplicate";
  std::string_view view1(source);
  std::string_view view2(source);

  unsigned id1 = container[view1];
  unsigned id2 = container[view2];

  REQUIRE(id1 == id2);
}

TEST_CASE("mixed types with string_view", "[string_view][mixed]")
{
  string_containert container;

  const char *c_str = "mixed";
  std::string cpp_str = "mixed";
  std::string_view sv = "mixed";

  unsigned id1 = container[c_str];
  unsigned id2 = container[cpp_str];
  unsigned id3 = container[sv];

  REQUIRE(id1 == id2);
  REQUIRE(id2 == id3);
}

// ============================================================================
// Operator[] Tests
// ============================================================================

TEST_CASE("operator[]: char* variant", "[operator]")
{
  string_containert container;

  unsigned id1 = container["operator_test"];
  unsigned id2 = container["operator_test"];

  REQUIRE(id1 == id2);
}

TEST_CASE("operator[]: std::string variant", "[operator]")
{
  string_containert container;

  std::string key = "operator_test";
  unsigned id1 = container[key];
  unsigned id2 = container[key];

  REQUIRE(id1 == id2);
}

// ============================================================================
// Global Static Container Tests
// ============================================================================

TEST_CASE("global container: singleton pattern", "[global][static]")
{
  string_containert &cont1 = get_string_container();
  string_containert &cont2 = get_string_container();

  REQUIRE(&cont1 == &cont2); // Same instance
}

TEST_CASE("global container: persistence across calls", "[global][static]")
{
  unsigned id1 = get_string_container()["global_test"];
  unsigned id2 = get_string_container()["global_test"];

  REQUIRE(id1 == id2);
}

TEST_CASE("global container: multiple insertions", "[global][static]")
{
  get_string_container()["global_1"];
  get_string_container()["global_2"];
  unsigned id3 = get_string_container()["global_3"];

  REQUIRE(id3 >= 1);
  REQUIRE(
    std::string(get_string_container().c_str(id3)) == std::string("global_3"));
}

TEST_CASE(
  "global container: thread safety initialization",
  "[global][static][thread]")
{
  std::vector<std::thread> threads;
  std::set<string_containert *> instances;
  std::mutex mutex;

  for (int i = 0; i < 10; ++i)
  {
    threads.emplace_back(
      [&mutex, &instances]()
      {
        auto &inst = get_string_container();
        {
          std::lock_guard<std::mutex> lock(mutex);
          instances.insert(&inst);
        }
      });
  }

  for (auto &t : threads)
  {
    t.join();
  }

  REQUIRE(instances.size() == 1); // All threads got the same instance
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_CASE("edge case: very long strings", "[edge]")
{
  string_containert container;

  std::string long_str(10000, 'a');
  unsigned id = container[long_str];

  REQUIRE(container.c_str(id) == long_str);
}

TEST_CASE("edge case: special characters", "[edge]")
{
  string_containert container;

  std::string special = "!@#$%^&*()_+-=[]{}|;':\",./<>?";
  unsigned id = container[special];

  REQUIRE(container.c_str(id) == special);
}

TEST_CASE("edge case: whitespace strings", "[edge]")
{
  string_containert container;

  unsigned id1 = container[" "];
  unsigned id2 = container["  "];
  unsigned id3 = container["\t"];
  unsigned id4 = container["\n"];

  REQUIRE(id1 != id2);
  REQUIRE(id2 != id3);
  REQUIRE(id3 != id4);
}

TEST_CASE("edge case: unicode characters", "[edge]")
{
  string_containert container;

  std::string unicode = "こんにちは世界🌍";
  unsigned id = container[unicode];

  REQUIRE(container.c_str(id) == unicode);
}

TEST_CASE("edge case: null character handling", "[edge]")
{
  string_containert container;

  // Note: null-terminated strings will stop at \0
  // This tests the behavior with embedded nulls (if supported)
  const char *str = "normal";
  unsigned id = container[str];

  REQUIRE(std::string(container.c_str(id)) == "normal");
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_CASE("stress: many unique strings", "[stress]")
{
  string_containert container;

  const int NUM_STRINGS = 10000;
  std::vector<unsigned> ids;

  for (int i = 0; i < NUM_STRINGS; ++i)
  {
    unsigned id = container["string_" + std::to_string(i)];
    ids.push_back(id);
  }

  // All IDs should be unique
  std::set<unsigned> unique_ids(ids.begin(), ids.end());
  REQUIRE(unique_ids.size() == NUM_STRINGS);

  // Verify all can be retrieved
  for (int i = 0; i < NUM_STRINGS; ++i)
  {
    REQUIRE(container.c_str(ids[i]) == ("string_" + std::to_string(i)));
  }
}

TEST_CASE("stress: repeated lookups", "[stress]")
{
  string_containert container;

  unsigned id = container["repeated"];

  // Lookup the same string many times
  for (int i = 0; i < 100000; ++i)
  {
    unsigned new_id = container["repeated"];
    REQUIRE(new_id == id);
  }
}

// ============================================================================
// Concurrency Tests
// ============================================================================

TEST_CASE(
  "concurrency: multiple threads inserting different strings",
  "[concurrency][thread]")
{
  string_containert container;
  std::vector<std::thread> threads;
  std::vector<unsigned> ids(10);
  std::mutex mutex;

  for (int i = 0; i < 10; ++i)
  {
    threads.emplace_back(
      [&, i]()
      {
        std::string str = "thread_" + std::to_string(i);
        unsigned id = container[str];
        {
          std::lock_guard<std::mutex> lock(mutex);
          ids[i] = id;
        }
      });
  }

  for (auto &t : threads)
  {
    t.join();
  }

  // All IDs should be unique
  std::set<unsigned> unique_ids(ids.begin(), ids.end());
  REQUIRE(unique_ids.size() == 10);
}

TEST_CASE(
  "concurrency: multiple threads looking up same string",
  "[concurrency][thread]")
{
  string_containert container;
  unsigned master_id = container["shared"];

  std::vector<unsigned> retrieved_ids(100);
  std::vector<std::thread> threads;
  std::mutex mutex;

  for (int i = 0; i < 100; ++i)
  {
    threads.emplace_back(
      [&, i]()
      {
        unsigned id = container["shared"];
        {
          std::lock_guard<std::mutex> lock(mutex);
          retrieved_ids[i] = id;
        }
      });
  }

  for (auto &t : threads)
  {
    t.join();
  }

  // All should get the same ID
  for (unsigned id : retrieved_ids)
  {
    REQUIRE(id == master_id);
  }
}

TEST_CASE(
  "concurrency: stress test mixed operations",
  "[concurrency][thread][stress]")
{
  string_containert container;
  const int NUM_THREADS = 10;
  const int STRINGS_PER_THREAD = 1000;

  std::vector<std::thread> threads;
  std::set<unsigned> all_ids;
  std::mutex mutex;

  for (int t = 0; t < NUM_THREADS; ++t)
  {
    threads.emplace_back(
      [&, t]()
      {
        for (int i = 0; i < STRINGS_PER_THREAD; ++i)
        {
          std::string str =
            "thread_" + std::to_string(t) + "_str_" + std::to_string(i);
          unsigned id = container[str];

          // Also do some lookups
          for (int j = 0; j < 10; ++j)
          {
            unsigned lookup_id = container[str];
            REQUIRE(lookup_id == id);
          }

          {
            std::lock_guard<std::mutex> lock(mutex);
            all_ids.insert(id);
          }
        }
      });
  }

  for (auto &t : threads)
  {
    t.join();
  }

  // Should have NUM_THREADS * STRINGS_PER_THREAD unique IDs (plus empty string)
  REQUIRE(all_ids.size() == NUM_THREADS * STRINGS_PER_THREAD);
}

// ============================================================================
// Index Validity Tests
// ============================================================================

TEST_CASE("indices: sequential assignment", "[index]")
{
  string_containert container;

  // Empty string gets 0 automatically
  unsigned id0 = container[""];
  REQUIRE(id0 == 0);

  // Subsequent strings should get sequential indices
  unsigned id1 = container["first"];
  unsigned id2 = container["second"];
  unsigned id3 = container["third"];

  REQUIRE(id1 == 1);
  REQUIRE(id2 == 2);
  REQUIRE(id3 == 3);
}

TEST_CASE("indices: non-sequential after duplicates", "[index]")
{
  string_containert container;

  unsigned id1 = container["str1"];       // Gets index 1
  unsigned id1_again = container["str1"]; // Gets same index
  unsigned id2 = container["str2"];       // Gets index 2
  unsigned id1_third = container["str1"]; // Gets same index again
  unsigned id3 = container["str3"];       // Gets index 3

  REQUIRE(id1 == 1);
  REQUIRE(id1_again == 1);
  REQUIRE(id2 == 2);
  REQUIRE(id1_third == 1);
  REQUIRE(id3 == 3);
}
