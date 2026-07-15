// H-A9: the guard shared-prefix walk (common_pointer_prefix_size,
// irep2_guard.cpp:64-108) drives operator-= (called unconditionally at :342)
// and operator|= (:468). We verify the REAL guard2tc: build guards that share a
// cached and-chain prefix then diverge, run the real operators, and check the
// observable result against a naive reference. In asserts-on builds the real
// code additionally self-checks walk == scan (irep2_guard.cpp:106).

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <algorithm>
#include <string>
#include <vector>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_guard.h>
#include <irep2/irep2_utils.h>
#include <util/c_types.h>
#include <util/config.h>

namespace
{
expr2tc sym(const std::string &name)
{
  return symbol2tc(get_bool_type(), irep_idt(name));
}

// Order-independent identity of a guard's conjunct set (guard_list is a set).
std::vector<size_t> conjunct_crcs(const guard2tc &g)
{
  std::vector<size_t> v;
  for (const auto &c : g.guard_list)
    v.push_back(c->crc());
  std::sort(v.begin(), v.end());
  return v;
}

std::vector<size_t> crc_set(const std::vector<expr2tc> &v)
{
  std::vector<size_t> r;
  for (const auto &c : v)
    r.push_back(c->crc());
  std::sort(r.begin(), r.end());
  return r;
}

// Naive set difference g1 \ g2 by structural conjunct equality.
std::vector<expr2tc> naive_diff(const guard2tc &g1, const guard2tc &g2)
{
  std::vector<expr2tc> r;
  for (const auto &c : g1.guard_list)
  {
    bool in2 = false;
    for (const auto &d : g2.guard_list)
      if (c == d)
      {
        in2 = true;
        break;
      }
    if (!in2)
      r.push_back(c);
  }
  return r;
}

// Check g1 -= g2 against the naive set difference (mutates g1).
void check_minus_eq(guard2tc g1, const guard2tc &g2)
{
  std::vector<expr2tc> ref = naive_diff(g1, g2);
  g1 -= g2;
  if (ref.empty())
    REQUIRE(g1.is_true());
  else
  {
    REQUIRE(!g1.is_true());
    REQUIRE(conjunct_crcs(g1) == crc_set(ref));
  }
}
} // namespace

// operator-= over a shared cached prefix: both guards descend from the same
// base copy, so common_pointer_prefix_size walks the pointer-shared and-chain.
TEST_CASE("guard -= over a shared cached prefix (H-A9)", "[core][irep2][guard]")
{
  config.ansi_c.word_size = 32;

  for (unsigned shared = 0; shared <= 4; ++shared)
  {
    guard2tc base;
    for (unsigned i = 0; i < shared; ++i)
      base.add(sym("s" + std::to_string(i)));

    SECTION(
      "siblings each add one distinct conjunct, shared=" +
      std::to_string(shared))
    {
      guard2tc g1 = base, g2 = base;
      g1.add(sym("a"));
      g2.add(sym("b"));
      check_minus_eq(g1, g2); // -> {a}
    }

    SECTION("g2 is a pointer-prefix of g1, shared=" + std::to_string(shared))
    {
      guard2tc g1 = base, g2 = base;
      g1.add(sym("x"));
      g1.add(sym("y"));
      check_minus_eq(g1, g2); // -> {x, y}
    }

    SECTION("equal guards cancel to true, shared=" + std::to_string(shared))
    {
      guard2tc g1 = base, g2 = base;
      g1.add(sym("z"));
      g2.add(sym("z"));
      check_minus_eq(g1, g2); // shared prefix + same conjunct -> true
    }
  }
}

// operator-= with no shared prefix falls through common_pointer_prefix_size == 0
// into the order-independent hash-set path.
TEST_CASE("guard -= with no common prefix (H-A9)", "[core][irep2][guard]")
{
  config.ansi_c.word_size = 32;

  guard2tc g1, g2;
  g1.add(sym("p"));
  g1.add(sym("q"));
  g2.add(sym("q"));
  g2.add(sym("r"));
  check_minus_eq(g1, g2); // {p, q} \ {q, r} = {p}
}

// operator|= factors the shared prefix out and disjoins the residuals; the
// prefix length must come out of common_pointer_prefix_size exactly.
TEST_CASE("guard |= factors the shared prefix (H-A9)", "[core][irep2][guard]")
{
  config.ansi_c.word_size = 32;

  for (unsigned shared = 1; shared <= 4; ++shared)
  {
    guard2tc base;
    std::vector<expr2tc> shared_atoms;
    for (unsigned i = 0; i < shared; ++i)
    {
      expr2tc a = sym("s" + std::to_string(i));
      shared_atoms.push_back(a);
      base.add(a);
    }

    guard2tc g1 = base, g2 = base;
    g1.add(sym("a"));
    g2.add(sym("b"));
    g1 |= g2; // (s0 & .. & s_{shared-1}) & (a | b)

    REQUIRE(g1.guard_list.size() == shared + 1);
    std::vector<expr2tc> prefix(
      g1.guard_list.begin(), g1.guard_list.begin() + shared);
    REQUIRE(crc_set(prefix) == crc_set(shared_atoms));
    REQUIRE(is_or2t(g1.guard_list[shared]));
  }
}

// g2 a prefix of g1 means g1 => g2, so g1 |= g2 collapses to g2. This case
// returns early via cached_prefix_expr and does NOT reach
// common_pointer_prefix_size; it pins the algebra, not the H-A9 walk itself.
TEST_CASE(
  "guard |= subsumes when g2 is a prefix (H-A9)",
  "[core][irep2][guard]")
{
  config.ansi_c.word_size = 32;

  guard2tc g2;
  g2.add(sym("s0"));
  g2.add(sym("s1"));
  guard2tc g1 = g2;
  g1.add(sym("a"));

  g1 |= g2;
  REQUIRE(conjunct_crcs(g1) == conjunct_crcs(g2));
}

// Deterministic sweep: many shared-prefix guard pairs with mixed suffixes,
// each -= cross-checked against the naive set difference. Broad coverage of
// common_pointer_prefix_size on real chains without any wall-clock/random dep.
TEST_CASE("guard -= sweep vs naive difference (H-A9)", "[core][irep2][guard]")
{
  config.ansi_c.word_size = 32;

  const char *pool[] = {"w", "x", "y", "z"};
  for (uint32_t seed = 1; seed <= 200; ++seed)
  {
    uint32_t r = seed;
    auto nextbits = [&r]() {
      r = r * 1103515245u + 12345u;
      return r >> 16;
    };

    unsigned shared = nextbits() % 3; // 0..2 shared leading conjuncts
    guard2tc base;
    for (unsigned i = 0; i < shared; ++i)
      base.add(sym("s" + std::to_string(i)));

    guard2tc g1 = base, g2 = base;
    unsigned m1 = nextbits() & 0xf, m2 = nextbits() & 0xf;
    for (unsigned i = 0; i < 4; ++i)
      if (m1 & (1u << i))
        g1.add(sym(pool[i]));
    for (unsigned i = 0; i < 4; ++i)
      if (m2 & (1u << i))
        g2.add(sym(pool[i]));

    check_minus_eq(g1, g2);
    check_minus_eq(g2, g1);
  }
}
