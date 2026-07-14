// H-B4: guard set-algebra logical equivalence. operator-= and operator|= on the
// REAL guard2tc are heavily optimised (cached-prefix fast paths, pointer-keyed
// set ops); a wrong result is an unsound path condition. We verify the genuine
// operators against naive references by evaluating the real as_expr() result
// trees exhaustively over every boolean assignment of their atoms — an exact
// equivalence check on boolean terms, no external SMT needed.
//
// Oracles per the plan (§H-B4):
//   -=  conjunct SET DIFFERENCE  (NOT g1 & ~shared): result == AND(g1 \ g2)
//   |=  logical OR:              as_expr(g1 |= g2) <=> as_expr(g1) | as_expr(g2)

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <cstdint>
#include <map>
#include <set>
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

// Exact boolean evaluation of a guard as_expr() tree under an assignment.
bool eval(const expr2tc &e, const std::map<std::string, bool> &asn)
{
  if (is_constant_bool2t(e))
    return to_constant_bool2t(e).value;
  if (is_symbol2t(e))
    return asn.at(to_symbol2t(e).thename.as_string());
  if (is_not2t(e))
    return !eval(to_not2t(e).value, asn);
  if (is_and2t(e))
    return eval(to_and2t(e).side_1, asn) && eval(to_and2t(e).side_2, asn);
  if (is_or2t(e))
    return eval(to_or2t(e).side_1, asn) || eval(to_or2t(e).side_2, asn);
  FAIL("unexpected expr kind in guard as_expr");
  return false;
}

void collect_atoms(const expr2tc &e, std::set<std::string> &out)
{
  if (is_symbol2t(e))
    out.insert(to_symbol2t(e).thename.as_string());
  else if (is_not2t(e))
    collect_atoms(to_not2t(e).value, out);
  else if (is_and2t(e))
  {
    collect_atoms(to_and2t(e).side_1, out);
    collect_atoms(to_and2t(e).side_2, out);
  }
  else if (is_or2t(e))
  {
    collect_atoms(to_or2t(e).side_1, out);
    collect_atoms(to_or2t(e).side_2, out);
  }
}

// A and B agree on every 2^n boolean assignment of the atoms they mention.
void require_equiv(const expr2tc &A, const expr2tc &B)
{
  std::set<std::string> names;
  collect_atoms(A, names);
  collect_atoms(B, names);
  std::vector<std::string> atoms(names.begin(), names.end());
  REQUIRE(atoms.size() <= 16); // bounds the 1u << n truth-table enumeration

  for (uint32_t mask = 0; mask < (1u << atoms.size()); ++mask)
  {
    std::map<std::string, bool> asn;
    for (size_t i = 0; i < atoms.size(); ++i)
      asn[atoms[i]] = (mask >> i) & 1u;
    REQUIRE(eval(A, asn) == eval(B, asn));
  }
}

// naive AND of g1's conjuncts absent (structurally) from g2.
expr2tc naive_diff_expr(const guard2tc &g1, const guard2tc &g2)
{
  std::vector<expr2tc> diff;
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
      diff.push_back(c);
  }
  return conjunction(diff);
}

// g1 -= g2 must equal AND(g1 \ g2) (set difference), logically.
void check_minus(guard2tc g1, const guard2tc &g2)
{
  expr2tc want = naive_diff_expr(g1, g2);
  g1 -= g2;
  require_equiv(g1.as_expr(), want);
}

// g1 |= g2 must equal as_expr(g1) | as_expr(g2), logically.
void check_or(guard2tc g1, const guard2tc &g2)
{
  expr2tc want = or2tc(g1.as_expr(), g2.as_expr());
  g1 |= g2;
  require_equiv(g1.as_expr(), want);
}
} // namespace

TEST_CASE("guard -= is conjunct set difference (H-B4)", "[core][irep2][guard]")
{
  config.ansi_c.word_size = 32;

  SECTION("shared cached prefix, divergent suffix")
  {
    guard2tc base;
    base.add(sym("s0"));
    base.add(sym("s1"));
    guard2tc g1 = base, g2 = base;
    g1.add(sym("a"));
    g2.add(sym("b"));
    check_minus(g1, g2); // -> a
  }

  SECTION("g2 is a prefix of g1")
  {
    guard2tc g2;
    g2.add(sym("s0"));
    g2.add(sym("s1"));
    guard2tc g1 = g2;
    g1.add(sym("x"));
    check_minus(g1, g2); // -> x
  }

  SECTION("overlapping suffix conjuncts")
  {
    guard2tc g1, g2;
    for (const char *n : {"p", "q", "r"})
      g1.add(sym(n));
    for (const char *n : {"q", "r", "t"})
      g2.add(sym(n));
    check_minus(g1, g2); // {p,q,r} \ {q,r,t} = p
  }

  SECTION("equal guards cancel")
  {
    guard2tc g1, g2;
    g1.add(sym("a"));
    g1.add(sym("b"));
    g2.add(sym("a"));
    g2.add(sym("b"));
    check_minus(g1, g2); // -> true
  }
}

TEST_CASE("guard |= is logical disjunction (H-B4)", "[core][irep2][guard]")
{
  config.ansi_c.word_size = 32;

  SECTION("shared prefix siblings")
  {
    guard2tc base;
    base.add(sym("s0"));
    base.add(sym("s1"));
    guard2tc g1 = base, g2 = base;
    g1.add(sym("a"));
    g2.add(sym("b"));
    check_or(g1, g2); // (s0&s1) & (a|b)
  }

  SECTION("g2 prefix of g1 subsumes")
  {
    guard2tc g2;
    g2.add(sym("s0"));
    g2.add(sym("s1"));
    guard2tc g1 = g2;
    g1.add(sym("a"));
    check_or(g1, g2); // -> s0&s1
  }

  SECTION("disjoint guards")
  {
    guard2tc g1, g2;
    g1.add(sym("a"));
    g1.add(sym("b"));
    g2.add(sym("c"));
    check_or(g1, g2);
  }

  SECTION("complementary residuals simplify to true")
  {
    guard2tc base;
    base.add(sym("s0"));
    guard2tc g1 = base, g2 = base;
    g1.add(sym("a"));
    g2.add(not2tc(sym("a")));
    check_or(g1, g2); // (s0 & (a | !a)) == s0
  }
}

// Deterministic sweep: guard pairs over a 4-atom pool (optionally sharing a
// leading conjunct), each cross-checked for both operators. 2^n truth tables
// stay tiny; no wall-clock or random dependency.
TEST_CASE("guard algebra sweep (H-B4)", "[core][irep2][guard]")
{
  config.ansi_c.word_size = 32;

  const char *pool[] = {"a", "b", "c", "d"};
  for (uint32_t seed = 1; seed <= 150; ++seed)
  {
    uint32_t r = seed;
    auto nextbits = [&r]() {
      r = r * 1103515245u + 12345u;
      return r >> 16;
    };

    guard2tc base;
    if (nextbits() & 1u)
      base.add(sym("s0")); // sometimes a shared cached prefix

    guard2tc g1 = base, g2 = base;
    unsigned m1 = nextbits() & 0xf, m2 = nextbits() & 0xf;
    for (unsigned i = 0; i < 4; ++i)
      if (m1 & (1u << i))
        g1.add(sym(pool[i]));
    for (unsigned i = 0; i < 4; ++i)
      if (m2 & (1u << i))
        g2.add(sym(pool[i]));

    check_minus(g1, g2);
    check_minus(g2, g1);
    check_or(g1, g2);
    check_or(g2, g1);
  }
}
