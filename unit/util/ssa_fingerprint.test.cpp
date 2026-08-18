#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/ssa/fingerprint.h>

namespace
{
expr2tc local(const std::string &name, unsigned offset, unsigned version)
{
  return symbol2tc(
    get_int32_type(),
    irep_idt("c:t.c@" + std::to_string(offset) + "@F@main@" + name),
    symbol_renaming_level::level2,
    0,
    version,
    0,
    0);
}

void add_step(
  symex_target_equationt::SSA_stepst &steps,
  const expr2tc &lhs,
  int literal)
{
  steps.emplace_back();
  auto &s = steps.back();
  s.type = goto_trace_stept::ASSIGNMENT;
  s.guard = gen_true_expr();
  s.cond = equality2tc(lhs, constant_int2tc(get_int32_type(), BigInt(literal)));
}
} // namespace

TEST_CASE("step order does not change the digest", "[fingerprint]")
{
  symex_target_equationt::SSA_stepst forward, reversed;
  for (int i = 0; i < 4; ++i)
    add_step(forward, local("x", 100 + 10 * i, i), i);
  for (int i = 3; i >= 0; --i)
    add_step(reversed, local("x", 100 + 10 * i, i), i);

  // The steps are conjuncts; symex does not emit them in a stable order
  // across unrelated edits, so the canonical form must not depend on it.
  REQUIRE(
    ssa_cone_digest(forward, fingerprint_modet::srcloc) ==
    ssa_cone_digest(reversed, fingerprint_modet::srcloc));
}

TEST_CASE(
  "a symbol's source offset does not change the digest",
  "[fingerprint]")
{
  symex_target_equationt::SSA_stepst before, after;
  add_step(before, local("x", 100, 0), 7);
  // Same variable after an edit inserted text earlier in the file.
  add_step(after, local("x", 144, 0), 7);

  REQUIRE(
    ssa_cone_digest(before, fingerprint_modet::srcloc) ==
    ssa_cone_digest(after, fingerprint_modet::srcloc));
  REQUIRE(
    ssa_cone_digest(before, fingerprint_modet::raw) !=
    ssa_cone_digest(after, fingerprint_modet::raw));
}

TEST_CASE("distinct cones keep distinct digests", "[fingerprint]")
{
  symex_target_equationt::SSA_stepst one, other;
  add_step(one, local("x", 100, 0), 7);
  add_step(other, local("x", 100, 0), 8);

  REQUIRE(
    ssa_cone_digest(one, fingerprint_modet::srcloc) !=
    ssa_cone_digest(other, fingerprint_modet::srcloc));
}

TEST_CASE("ignored steps are excluded", "[fingerprint]")
{
  symex_target_equationt::SSA_stepst kept, with_ignored;
  add_step(kept, local("x", 100, 0), 7);
  add_step(with_ignored, local("x", 100, 0), 7);
  add_step(with_ignored, local("y", 200, 0), 9);
  with_ignored.back().ignore = true;

  REQUIRE(ssa_cone_size(kept) == 1);
  REQUIRE(ssa_cone_size(with_ignored) == 1);
  REQUIRE(
    ssa_cone_digest(kept, fingerprint_modet::srcloc) ==
    ssa_cone_digest(with_ignored, fingerprint_modet::srcloc));
}
