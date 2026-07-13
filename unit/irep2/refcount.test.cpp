// H-A1: refcount conservation, single-free and self-alias safety of the real
// irep_container, checked by reading irep2t::refcount after each operation.

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <utility>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/config.h>

#include "refcount_ops.h"

namespace
{
// const get() reads the node without detaching, so it does not perturb the
// refcount being measured.
unsigned live_refcount(const expr2tc &c)
{
  return c.get()->refcount.load(std::memory_order_relaxed);
}

const expr2t *node_of(const expr2tc &c)
{
  return c.get();
}
} // namespace

TEST_CASE("irep2 refcount tracks live handles (I1)", "[core][irep2][refcount]")
{
  config.ansi_c.word_size = 32;

  expr2tc a = gen_ulong(11); // make_irep adopts at refcount 1
  REQUIRE(live_refcount(a) == 1);

  {
    expr2tc b = a; // copy ctor: fetch_add -> 2
    REQUIRE(live_refcount(a) == 2);
    REQUIRE(node_of(a) == node_of(b)); // shared, same node

    {
      expr2tc c = a; // -> 3
      REQUIRE(live_refcount(a) == 3);

      expr2tc d = std::move(c); // move ctor: steals, NO bump
      REQUIRE(live_refcount(a) == 3);
      REQUIRE(!c);                       // moved-from container is empty
      REQUIRE(node_of(d) == node_of(a)); // d owns the stolen node
    } // d destroyed -> 2
    REQUIRE(live_refcount(a) == 2);
  } // b destroyed -> 1
  REQUIRE(live_refcount(a) == 1);
}

// Copy-assign bumps the source and releases the previous pointee.
TEST_CASE(
  "irep2 copy-assign releases old and bumps new",
  "[core][irep2][refcount]")
{
  config.ansi_c.word_size = 32;

  expr2tc a = gen_ulong(7);  // node A, refcount 1
  expr2tc b = gen_ulong(99); // node B, refcount 1
  const expr2t *A = node_of(a);
  REQUIRE(node_of(b) != A);

  b = a; // release B (freed), fetch_add on A -> 2

  REQUIRE(live_refcount(a) == 2);
  REQUIRE(node_of(b) == A); // b now aliases A
  REQUIRE(to_constant_int2t(b).value.to_uint64() == 7);
}

// H-A3: assigning a container from a member of the node it solely owns must
// not use-after-free (operator= snapshots+bumps the source before release()).
// Mirrors the `x = to_array_type(x).subtype` hazard from irep2.h.
TEST_CASE(
  "irep2 self-aliasing assignment is UAF-safe (H-A3)",
  "[core][irep2][refcount]")
{
  config.ansi_c.word_size = 32;
  type2tc word = get_uint_type(config.ansi_c.word_size);

  expr2tc x = add2tc(word, gen_ulong(3), gen_ulong(4)); // sole owner, rc 1
  REQUIRE(live_refcount(x) == 1);

  // RHS aliases a member (side_1) of the node x solely owns.
  const expr2tc &alias = to_add2t(x).side_1;
  x = alias; // must not read freed storage; x becomes the operand (== 3)

  REQUIRE(is_constant_int2t(x));
  REQUIRE(to_constant_int2t(x).value.to_uint64() == 3);
  REQUIRE(live_refcount(x) == 1);
}

// COW detach on a shared node clones, leaving the other handle sole owner.
TEST_CASE(
  "irep2 COW detach conserves refcount and isolates mutation",
  "[core][irep2][refcount]")
{
  config.ansi_c.word_size = 32;

  expr2tc a = gen_ulong(11);
  expr2tc b = a; // shared, refcount 2
  REQUIRE(live_refcount(a) == 2);
  REQUIRE(node_of(a) == node_of(b));

  (void)a.get(); // non-const get() -> detach(): a clones to a fresh node

  REQUIRE(node_of(a) != node_of(b)); // distinct nodes now
  REQUIRE(live_refcount(a) == 1);    // a sole-owns its clone
  REQUIRE(live_refcount(b) == 1);    // b sole-owns the original
  REQUIRE(*a == *b);                 // structural value unchanged
}

// run_ops over deterministic inputs: many operation sequences on real
// containers, each checked for refcount conservation (I1).
TEST_CASE(
  "irep2 refcount conservation under nondet operation sequences",
  "[core][irep2][refcount]")
{
  using irep2_refcount_fuzz::run_ops;

  SECTION("hand-crafted sequences exercising each opcode")
  {
    // {sel, arg} pairs: sel low 3 bits = op, high bits = slot i; arg = slot j.
    const uint8_t seqs[][8] = {
      {0x00, 7, 0x09, 0, 0x11, 0, 0x1c, 0},
      {0x00, 3, 0x0a, 0, 0x02, 0, 0x0d, 0},
      {0x00, 9, 0x0e, 0, 0x16, 1, 0x1d, 2},
      {0x00, 1, 0x09, 0, 0x0a, 0, 0x22, 1},
    };
    for (const auto &s : seqs)
      REQUIRE(run_ops(s, sizeof(s)));
  }

  SECTION("fixed-seed pseudo-random op streams (deterministic)")
  {
    // LCG gives reproducible coverage with no wall-clock/random dependency.
    for (uint32_t seed = 1; seed <= 64; ++seed)
    {
      uint8_t buf[512];
      uint32_t x = seed;
      for (uint8_t &b : buf)
      {
        x = x * 1103515245u + 12345u; // glibc LCG constants
        b = static_cast<uint8_t>(x >> 16);
      }
      REQUIRE(run_ops(buf, sizeof(buf)));
    }
  }
}
