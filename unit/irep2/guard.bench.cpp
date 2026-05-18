#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_guard.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/config.h>

namespace
{
expr2tc sym(const std::string &name)
{
  return symbol2tc(get_bool_type(), irep_idt(name));
}

// Build a guard of `n` distinct bool symbol conjuncts by repeated add().
// Matches the production hot path: long path-condition guards are grown
// one conjunct at a time as symex descends.
guard2tc build_guard(unsigned n, const std::string &prefix = "g")
{
  guard2tc g;
  for (unsigned i = 0; i < n; ++i)
    g.add(sym(prefix + std::to_string(i)));
  return g;
}

// Build two guards that share `shared` leading conjuncts then diverge.
// Used to exercise operator|='s set-intersection/difference factoring.
std::pair<guard2tc, guard2tc>
build_overlapping(unsigned shared, unsigned diverge)
{
  guard2tc base = build_guard(shared, "s");
  guard2tc a = base;
  guard2tc b = base;
  for (unsigned i = 0; i < diverge; ++i)
    a.add(sym("a" + std::to_string(i)));
  for (unsigned i = 0; i < diverge; ++i)
    b.add(sym("b" + std::to_string(i)));
  return {a, b};
}
} // namespace

TEST_CASE("guard2tc microbench: incremental construction", "[bench]")
{
  config.ansi_c.word_size = 32;

  BENCHMARK("add() x 16")
  {
    return build_guard(16);
  };
  BENCHMARK("add() x 64")
  {
    return build_guard(64);
  };
  BENCHMARK("add() x 256")
  {
    return build_guard(256);
  };
}

TEST_CASE("guard2tc microbench: copy", "[bench]")
{
  config.ansi_c.word_size = 32;
  guard2tc g16 = build_guard(16);
  guard2tc g64 = build_guard(64);
  guard2tc g256 = build_guard(256);

  BENCHMARK("copy guard size=16")
  {
    guard2tc copy = g16;
    return copy;
  };
  BENCHMARK("copy guard size=64")
  {
    guard2tc copy = g64;
    return copy;
  };
  BENCHMARK("copy guard size=256")
  {
    guard2tc copy = g256;
    return copy;
  };
}

TEST_CASE("guard2tc microbench: as_expr", "[bench]")
{
  config.ansi_c.word_size = 32;
  guard2tc empty;
  guard2tc one = build_guard(1);
  guard2tc g16 = build_guard(16);
  guard2tc g256 = build_guard(256);

  BENCHMARK("as_expr() empty")
  {
    return empty.as_expr();
  };
  BENCHMARK("as_expr() size=1")
  {
    return one.as_expr();
  };
  BENCHMARK("as_expr() size=16")
  {
    return g16.as_expr();
  };
  BENCHMARK("as_expr() size=256")
  {
    return g256.as_expr();
  };
}

TEST_CASE("guard2tc microbench: equality", "[bench]")
{
  config.ansi_c.word_size = 32;
  guard2tc a16 = build_guard(16);
  guard2tc b16 = build_guard(16);
  guard2tc a64 = build_guard(64);
  guard2tc b64 = build_guard(64);
  guard2tc a256 = build_guard(256);
  guard2tc b256 = build_guard(256);
  guard2tc differ16 = build_guard(16, "h");
  guard2tc differ256 = build_guard(256, "h");

  // Pre-warm the crc cache on a separate set so we can compare cold vs
  // warm code paths. We hash the underlying expr2tc base, not the
  // guard2tc itself, because the crc cache lives on the irep2 node.
  guard2tc warm_a256 = build_guard(256);
  guard2tc warm_b256 = build_guard(256);
  guard2tc warm_differ256 = build_guard(256, "h");
  (void)static_cast<const expr2tc &>(warm_a256)->crc();
  (void)static_cast<const expr2tc &>(warm_b256)->crc();
  (void)static_cast<const expr2tc &>(warm_differ256)->crc();

  BENCHMARK("equal size=16")
  {
    return a16 == b16;
  };
  BENCHMARK("equal size=64")
  {
    return a64 == b64;
  };
  BENCHMARK("equal size=256")
  {
    return a256 == b256;
  };
  BENCHMARK("unequal size=16 (full-list cmp)")
  {
    return a16 == differ16;
  };
  BENCHMARK("unequal size=256 (cold crc, full-list cmp)")
  {
    return a256 == differ256;
  };
  BENCHMARK("unequal size=256 (warm crc, fast path)")
  {
    return warm_a256 == warm_differ256;
  };
  BENCHMARK("equal size=256 (warm crc)")
  {
    return warm_a256 == warm_b256;
  };
}

TEST_CASE("guard2tc microbench: operator|=", "[bench]")
{
  config.ansi_c.word_size = 32;

  BENCHMARK("|= shared=8 diverge=8")
  {
    auto [a, b] = build_overlapping(8, 8);
    a |= b;
    return a;
  };
  BENCHMARK("|= shared=32 diverge=32")
  {
    auto [a, b] = build_overlapping(32, 32);
    a |= b;
    return a;
  };
  BENCHMARK("|= shared=128 diverge=32")
  {
    auto [a, b] = build_overlapping(128, 32);
    a |= b;
    return a;
  };
}

TEST_CASE("guard2tc microbench: operator-=", "[bench]")
{
  config.ansi_c.word_size = 32;

  BENCHMARK("-= small (16 minus shared 8)")
  {
    guard2tc shared = build_guard(8, "s");
    guard2tc a = shared;
    for (unsigned i = 0; i < 8; ++i)
      a.add(sym("a" + std::to_string(i)));
    a -= shared;
    return a;
  };
  BENCHMARK("-= big (256 minus shared 128)")
  {
    guard2tc shared = build_guard(128, "s");
    guard2tc a = shared;
    for (unsigned i = 0; i < 128; ++i)
      a.add(sym("a" + std::to_string(i)));
    a -= shared;
    return a;
  };
}
