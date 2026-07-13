// H-B2: CRC must be consistent with structural equality (invariant I4):
//   a == b  =>  a.crc() == b.crc()   (no false CRC splits — mandatory)
// and CRC must be a deterministic function of structure (independently built,
// structurally identical trees hash identically). The reverse (a != b =>
// distinct crc) is only probabilistic, so collisions are *reported*, not
// asserted. All checks run against the REAL expr2t/type2t crc(), over a corpus
// with structurally-equal-but-distinct-pointer pairs so the crc path (not the
// same-pointer short-circuit) is exercised.

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <vector>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/c_types.h>
#include <util/config.h>

namespace
{
// a == b => a.crc() == b.crc(); count (but don't fail on) crc collisions.
template <typename C>
void check_crc_consistency(const std::vector<C> &corpus)
{
  size_t distinct_pairs = 0, collisions = 0;
  for (size_t i = 0; i < corpus.size(); ++i)
    for (size_t j = 0; j < corpus.size(); ++j)
    {
      const bool eq = (corpus[i] == corpus[j]);
      const size_t ci = corpus[i]->crc();
      const size_t cj = corpus[j]->crc();
      if (eq)
        REQUIRE(ci == cj); // I4: equal nodes never hash apart
      else
      {
        ++distinct_pairs;
        if (ci == cj)
          ++collisions;
      }
    }
  if (collisions)
    WARN(
      "crc collisions on distinct nodes: " << collisions << "/"
                                           << distinct_pairs);
}

expr2tc deep_add_chain(unsigned depth)
{
  type2tc u32 = get_uint_type(32);
  expr2tc acc = constant_int2tc(u32, BigInt(0));
  for (unsigned i = 0; i < depth; ++i)
    acc = add2tc(u32, acc, constant_int2tc(u32, BigInt(i)));
  return acc;
}
} // namespace

TEST_CASE("expr crc is consistent with equality (H-B2)", "[core][irep2]")
{
  config.ansi_c.word_size = 32;

  type2tc u32 = get_uint_type(32);
  expr2tc c5 = constant_int2tc(u32, BigInt(5));
  expr2tc c7 = constant_int2tc(u32, BigInt(7));

  std::vector<expr2tc> corpus{
    gen_true_expr(),
    gen_false_expr(),
    constant_int2tc(u32, BigInt(5)),
    constant_int2tc(u32, BigInt(5)), // equal to [2], distinct pointer
    constant_int2tc(u32, BigInt(7)),
    constant_int2tc(get_int_type(64), BigInt(-5)),
    constant_int2tc(get_uint_type(8), BigInt(5)), // same value, other width
    symbol2tc(u32, "x"),
    symbol2tc(u32, "x"), // equal to [7], distinct pointer
    symbol2tc(u32, "y"),
    add2tc(u32, c5, c7),
    add2tc(u32, c5, c7), // equal to [10], distinct pointer
  };

  check_crc_consistency(corpus);
}

TEST_CASE("type crc is consistent with equality (H-B2)", "[core][irep2]")
{
  config.ansi_c.word_size = 32;

  expr2tc sz2 = constant_int2tc(get_uint_type(32), BigInt(2));

  std::vector<type2tc> corpus{
    get_bool_type(),
    get_uint_type(32),
    unsignedbv_type2tc(32), // equal to [1] but a fresh allocation
    get_uint_type(64),
    get_int_type(32),
    pointer_type2tc(get_uint_type(8)),
    array_type2tc(get_uint_type(8), sz2, false),
    array_type2tc(
      get_uint_type(8),
      constant_int2tc(get_uint_type(32), BigInt(2)),
      false), // equal to [6], independently built
  };

  check_crc_consistency(corpus);
}

// CRC is a deterministic function of structure: two independently constructed,
// structurally identical trees hash identically (stronger than cold==warm on a
// single object, which irep2.test.cpp already covers).
TEST_CASE("crc is deterministic across construction (H-B2)", "[core][irep2]")
{
  config.ansi_c.word_size = 32;

  expr2tc a = deep_add_chain(2000);
  expr2tc b = deep_add_chain(2000);
  REQUIRE(a == b);
  REQUIRE(a->crc() == b->crc());

  // A one-element-shorter chain is a different structure and must not collide
  // with the full one (sanity that the chain length is actually mixed in).
  expr2tc shorter = deep_add_chain(1999);
  REQUIRE(a != shorter);
  REQUIRE(a->crc() != shorter->crc());
}
