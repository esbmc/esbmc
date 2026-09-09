/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unit tests for persistent_map / persistent_set: the O(1)-copy structures
 * backing the symex renaming maps, value set, and frame local-variable set.
 * The immutable-snapshot property tested here is what makes a saved branch
 * state safe to keep while the live path mutates its own copy, and the diff()
 * behaviour is what the value-set merge, phi_function, and merge_locality use
 * to reconcile two paths in O(divergence).
 */

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <util/persistent_map.h>

#include <functional>
#include <vector>

namespace
{
using imap = persistent_map<unsigned, int, std::hash<unsigned>>;
using iset = persistent_set<unsigned, std::hash<unsigned>>;
} // namespace

TEST_CASE("persistent_map set / find / at / erase", "[util][persistent-map]")
{
  imap m;
  REQUIRE(m.empty());

  m.set(1, 10);
  m.set(2, 20);
  REQUIRE(m.size() == 2);
  REQUIRE_FALSE(m.empty());

  REQUIRE(m.find(1) != nullptr);
  REQUIRE(*m.find(1) == 10);
  REQUIRE(m.at(2) == 20);
  REQUIRE(m.find(3) == nullptr);

  m.set(1, 11); // overwrite keeps one entry
  REQUIRE(m.at(1) == 11);
  REQUIRE(m.size() == 2);

  REQUIRE(m.erase(1));       // present -> true
  REQUIRE_FALSE(m.erase(1)); // already gone -> false
  REQUIRE(m.find(1) == nullptr);
  REQUIRE(m.size() == 1);

  m.clear();
  REQUIRE(m.empty());
}

TEST_CASE(
  "a persistent_map copy is an independent snapshot",
  "[util][persistent-map]")
{
  // The branch-snapshot optimisation depends on this: the saved merge state
  // must not observe mutations the live path makes to its own copy.
  imap original;
  original.set(1, 10);
  original.set(2, 20);

  imap snapshot = original;
  original.set(2, 99);
  original.set(3, 30);
  original.erase(1);

  REQUIRE(snapshot.at(1) == 10);
  REQUIRE(snapshot.at(2) == 20);
  REQUIRE(snapshot.find(3) == nullptr);
  REQUIRE(snapshot.size() == 2);
}

TEST_CASE(
  "persistent_map diff reports added, removed and changed keys",
  "[util][persistent-map]")
{
  imap a;
  a.set(1, 10);
  a.set(2, 20);
  a.set(3, 30);

  imap b = a;
  b.erase(1);   // in a, not in b  -> removed
  b.set(2, 22); // shared key, differing value -> changed
  b.set(4, 40); // in b, not in a  -> added

  std::vector<unsigned> added, removed, changed;
  a.diff(
    b,
    [&](const auto &kv) { added.push_back(kv.first); },
    [&](const auto &kv) { removed.push_back(kv.first); },
    [&](const auto &, const auto &kv) { changed.push_back(kv.first); });

  REQUIRE(added == std::vector<unsigned>{4});
  REQUIRE(removed == std::vector<unsigned>{1});
  REQUIRE(changed == std::vector<unsigned>{2});
}

TEST_CASE("persistent_set insert / count / erase", "[util][persistent-set]")
{
  iset s;
  REQUIRE(s.empty());

  s.insert(1);
  s.insert(2);
  s.insert(2); // idempotent
  REQUIRE(s.size() == 2);
  REQUIRE(s.count(1) == 1);
  REQUIRE(s.count(3) == 0);

  REQUIRE(s.erase(1));
  REQUIRE_FALSE(s.erase(1));
  REQUIRE(s.count(1) == 0);
}

TEST_CASE(
  "persistent_set diff drives the merge_locality union",
  "[util][persistent-set]")
{
  // Mirrors merge_locality: union the merged path's locals into this one by
  // collecting the keys diff() reports as added (present in src, not here)
  // and inserting them; keys only here are kept untouched.
  iset cur;
  cur.insert(1);
  cur.insert(2);

  iset src = cur;
  src.insert(3); // src declared one more local
  cur.insert(4); // this path declared a different one

  std::vector<unsigned> added;
  cur.diff(
    src, [&](const auto &k) { added.push_back(k); }, [](const auto &) {});
  for (unsigned k : added)
    cur.insert(k);

  REQUIRE(cur.count(1));
  REQUIRE(cur.count(2));
  REQUIRE(cur.count(3));
  REQUIRE(cur.count(4));
  REQUIRE(cur.size() == 4);

  // The snapshot the union read from stays untouched.
  REQUIRE(src.count(4) == 0);
  REQUIRE(src.size() == 3);
}
