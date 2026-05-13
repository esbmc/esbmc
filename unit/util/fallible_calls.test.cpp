/*******************************************************************\
Module: Unit tests for util/fallible_calls.h
\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <util/fallible_calls.h>

TEST_CASE("find_fallible matches canonical names", "[util][fallible_calls]")
{
  REQUIRE(find_fallible("calloc") != nullptr);
  REQUIRE(find_fallible("calloc")->kind == success_kind::non_null);
  REQUIRE(find_fallible("fopen")->kind == success_kind::non_null);
  REQUIRE(find_fallible("getenv")->kind == success_kind::non_null);
  REQUIRE(find_fallible("read")->kind == success_kind::non_negative);
  REQUIRE(find_fallible("recv")->kind == success_kind::non_negative);
  REQUIRE(find_fallible("pthread_mutex_lock")->kind == success_kind::zero);
}

TEST_CASE("find_fallible returns null for non-fallible names",
  "[util][fallible_calls]")
{
  REQUIRE(find_fallible("") == nullptr);
  REQUIRE(find_fallible("printf") == nullptr);
  REQUIRE(find_fallible("abort") == nullptr);
}

TEST_CASE(
  "find_fallible strips ESBMC operational-model suffixes",
  "[util][fallible_calls]")
{
  // pthread OM in src/c2goto/library/pthread_lib.c renames user-level
  // pthread_mutex_lock to pthread_mutex_lock_noassert; the lookup must
  // recover the canonical entry.
  const fallible_call_t *l = find_fallible("pthread_mutex_lock_noassert");
  REQUIRE(l != nullptr);
  REQUIRE(l->name == "pthread_mutex_lock");
  REQUIRE(l->kind == success_kind::zero);

  REQUIRE(find_fallible("pthread_mutex_unlock_nocheck") != nullptr);
  REQUIRE(find_fallible("pthread_mutex_unlock_nocheck")->name ==
          "pthread_mutex_unlock");

  // A non-pthread suffix must not match anything.
  REQUIRE(find_fallible("printf_noassert") == nullptr);
}

TEST_CASE("fallible_calls table is non-empty and free of duplicates",
  "[util][fallible_calls]")
{
  const auto &table = fallible_calls();
  REQUIRE_FALSE(table.empty());
  // Two entries must not share a name — find_fallible's first-match
  // semantics would otherwise silently mask the shadowed one.
  for (size_t i = 0; i < table.size(); ++i)
    for (size_t j = i + 1; j < table.size(); ++j)
      REQUIRE(table[i].name != table[j].name);
}
