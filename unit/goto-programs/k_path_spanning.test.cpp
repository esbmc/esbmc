/*******************************************************************
 Module: k-path coverage spanning-set unit tests (issue #4335 PR1)
 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <goto-programs/k_path_spanning.h>
#include <irep2/irep2.h>
#include <irep2/irep2_utils.h>

namespace
{
// Build a single atom from a fresh symbol expression. Uses the deep
// structural compare in atom_lt, so two `mk_atom("x", true)` calls
// produce atoms that compare equal regardless of pointer identity.
k_path_spanning_sett::atom_t mk_atom(const std::string &name, bool polarity)
{
  return {symbol2tc(get_int_type(32), name), polarity};
}
} // namespace

TEST_CASE(
  "k-path spanning-set: empty set has zero spanning size",
  "[goto-cover][k-path-spanning]")
{
  k_path_spanning_sett s;
  s.finalize();
  REQUIRE(s.total() == 0);
  REQUIRE(s.spanning_size() == 0);
}

TEST_CASE(
  "k-path spanning-set: a single goal is always maximal",
  "[goto-cover][k-path-spanning]")
{
  k_path_spanning_sett s;
  s.add_goal({mk_atom("x", true)}, "x", "loc1");
  s.finalize();
  REQUIRE(s.total() == 1);
  REQUIRE(s.spanning_size() == 1);
  REQUIRE_FALSE(s.is_redundant("x", "loc1"));
}

TEST_CASE(
  "k-path spanning-set: a proper-subset goal is dropped from the spanning set",
  "[goto-cover][k-path-spanning]")
{
  // {x} ⊂ {x, y} — the singleton is subsumed by the pair, so it should
  // not contribute to the spanning denominator.
  k_path_spanning_sett s;
  s.add_goal({mk_atom("x", true)}, "x", "loc1");
  s.add_goal({mk_atom("x", true), mk_atom("y", true)}, "x && y", "loc2");
  s.finalize();
  REQUIRE(s.total() == 2);
  REQUIRE(s.spanning_size() == 1);
  REQUIRE(s.is_redundant("x", "loc1"));
  REQUIRE_FALSE(s.is_redundant("x && y", "loc2"));
}

TEST_CASE(
  "k-path spanning-set: same-size atom multisets are mutually maximal",
  "[goto-cover][k-path-spanning]")
{
  // Two depth-2 goals differing only in the polarity of the second
  // atom — neither is a proper subset of the other.
  k_path_spanning_sett s;
  s.add_goal({mk_atom("x", true), mk_atom("y", true)}, "x && y", "loc1");
  s.add_goal({mk_atom("x", true), mk_atom("y", false)}, "x && !y", "loc2");
  s.finalize();
  REQUIRE(s.spanning_size() == 2);
  REQUIRE_FALSE(s.is_redundant("x && y", "loc1"));
  REQUIRE_FALSE(s.is_redundant("x && !y", "loc2"));
}

TEST_CASE(
  "k-path spanning-set: structural equality drops pointer-identity dependence",
  "[goto-cover][k-path-spanning]")
{
  // Construct the same `x` symbol twice as two distinct expr2tc values.
  // `mk_atom` makes a fresh symbol2tc each call, so the underlying
  // pointers differ, but irep2's deep compare must treat them as equal
  // — otherwise subsumption would silently fail across function passes.
  k_path_spanning_sett s;
  s.add_goal({mk_atom("x", true)}, "x", "loc1");
  s.add_goal({mk_atom("x", true), mk_atom("y", true)}, "x && y", "loc2");
  s.finalize();
  REQUIRE(s.is_redundant("x", "loc1"));
  REQUIRE_FALSE(s.is_redundant("x && y", "loc2"));
}

TEST_CASE(
  "k-path spanning-set: identical-polarity duplicates form a multiset",
  "[goto-cover][k-path-spanning]")
{
  // The {x} singleton must be a proper subset of {x, x} — duplicates
  // are preserved and counted, matching test 8's chain-contradiction
  // shape.
  k_path_spanning_sett s;
  s.add_goal({mk_atom("x", true)}, "x", "loc1");
  s.add_goal({mk_atom("x", true), mk_atom("x", true)}, "x && x", "loc2");
  s.finalize();
  REQUIRE(s.spanning_size() == 1);
  REQUIRE(s.is_redundant("x", "loc1"));
  REQUIRE_FALSE(s.is_redundant("x && x", "loc2"));
}

TEST_CASE(
  "k-path spanning-set: test-4 shape — 6 goals, 4 maximal",
  "[goto-cover][k-path-spanning]")
{
  // Replays the goal layout of regression test k_path_cov_4 (N=2, two
  // branches a>0 and taken):
  //   depth-1: {a},        {!a}                — both subsumed by depth-2
  //   depth-2: {a, t},     {a, !t},
  //            {!a, t},    {!a, !t}            — all 4 mutually maximal
  // Phase-1 would report 4/6 = 66.67%; Phase-2 spanning sees 4 maximal
  // goals as the denominator.
  k_path_spanning_sett s;
  s.add_goal({mk_atom("a", true)}, "a", "L1");
  s.add_goal({mk_atom("a", false)}, "!a", "L1");
  s.add_goal({mk_atom("a", true), mk_atom("t", true)}, "a && t", "L2");
  s.add_goal({mk_atom("a", true), mk_atom("t", false)}, "a && !t", "L2");
  s.add_goal({mk_atom("a", false), mk_atom("t", true)}, "!a && t", "L2");
  s.add_goal({mk_atom("a", false), mk_atom("t", false)}, "!a && !t", "L2");
  s.finalize();

  REQUIRE(s.total() == 6);
  REQUIRE(s.spanning_size() == 4);

  // Depth-1 emissions are spanning-set-redundant.
  REQUIRE(s.is_redundant("a", "L1"));
  REQUIRE(s.is_redundant("!a", "L1"));

  // Depth-2 emissions are feasible (maximal). This is the data path
  // bmc.cpp uses to populate the JSON `feasibility` field.
  REQUIRE_FALSE(s.is_redundant("a && t", "L2"));
  REQUIRE_FALSE(s.is_redundant("a && !t", "L2"));
  REQUIRE_FALSE(s.is_redundant("!a && t", "L2"));
  REQUIRE_FALSE(s.is_redundant("!a && !t", "L2"));
}

TEST_CASE(
  "k-path spanning-set: clear() resets state for re-use",
  "[goto-cover][k-path-spanning]")
{
  k_path_spanning_sett s;
  s.add_goal({mk_atom("x", true)}, "x", "loc1");
  s.add_goal({mk_atom("x", true), mk_atom("y", true)}, "x && y", "loc2");
  s.finalize();
  REQUIRE(s.spanning_size() == 1);

  s.clear();
  REQUIRE(s.total() == 0);
  REQUIRE(s.spanning_size() == 0);
  REQUIRE_FALSE(s.is_redundant("x", "loc1"));

  s.add_goal({mk_atom("z", true)}, "z", "loc3");
  s.finalize();
  REQUIRE(s.spanning_size() == 1);
  REQUIRE_FALSE(s.is_redundant("z", "loc3"));
}
