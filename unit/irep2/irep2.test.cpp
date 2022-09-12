#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <irep2/irep2.h>
#include <irep2/irep2_utils.h>
#include <util/crypto_hash.h>

namespace
{
std::array<unsigned int, 5> to_array(const crypto_hash &h)
{
  std::array<unsigned int, 5> result;
  std::copy(h.hash, h.hash + 5, result.begin());
  return result;
}
struct_type2tc testing_struct2t()
{
  std::vector<type2tc> struct_members{
    get_uint_type(config.ansi_c.word_size),
    get_uint_type(config.ansi_c.word_size)};
  std::vector<irep_idt> struct_memb_names{"a", "b"};
  return struct_type2tc(
    struct_members, struct_memb_names, struct_memb_names, "test_struct");
}

type2tc testing_overlap2t()
{
  return get_uint_type(1);
}

expr2tc gen_testing_overlap(unsigned v)
{
  return constant_int2tc(testing_overlap2t(), BigInt(v));
}

constant_struct2tc gen_testing_struct(unsigned int a, unsigned int b)
{
  std::vector<expr2tc> members{gen_ulong(a), gen_ulong(b)};
  return constant_struct2tc(testing_struct2t(), members);
}

void test_constructed_equally(const expr2tc e1, const expr2tc e2)
{
  crypto_hash c_hash;
  crypto_hash c_hash2;
  // "The == operator should return true"
  REQUIRE(e1 == e2);
  // "Their crc should be the same"
  REQUIRE(e1->do_crc() == e2->do_crc());
  // Their hash should be the same"
  e1->hash(c_hash);
  e2->hash(c_hash2);

  c_hash.fin();
  c_hash2.fin();
  REQUIRE(to_array(c_hash) == to_array(c_hash2));
  REQUIRE(c_hash.to_size_t() == c_hash2.to_size_t());
}

void test_constructed_differently(const expr2tc e1, const expr2tc e2)
{
  crypto_hash c_hash;
  crypto_hash c_hash2;

  // "The == operator should return false"
  REQUIRE(e1 != e2);
  // "Their crc should not be the same"
  REQUIRE(e1->do_crc() != e2->do_crc());
  // "Their hash should not be the same"

  e1->hash(c_hash);
  e2->hash(c_hash2);

  c_hash.fin();
  c_hash2.fin();
  REQUIRE(to_array(c_hash) != to_array(c_hash2));
  REQUIRE(c_hash.to_size_t() != c_hash2.to_size_t());
}

} // namespace

SCENARIO("irep2 hashing", "[core][irep2]")
{
  GIVEN("Expressions constructed in the same way")
  {
    std::vector<std::pair<expr2tc, expr2tc>> expressions{
      {gen_ulong(42), gen_ulong(42)},
      {gen_testing_struct(1, 2), gen_testing_struct(1, 2)}};

    THEN("Expressions and hashes should be equal to itself")
    {
      for(auto &e : expressions)
      {
        test_constructed_equally(e.first, e.first);
        test_constructed_equally(e.second, e.second);
      }
    }
    THEN("Expressions and hashes between pairs should be equal")
    {
      for(auto &e : expressions)
        test_constructed_equally(e.first, e.second);
    }
  }
  GIVEN("Expressions constructed differently")
  {
    std::vector<std::pair<expr2tc, expr2tc>> expressions{
      {gen_ulong(42), gen_ulong(64)},
      {gen_testing_struct(1, 2), gen_ulong(1)},
      {gen_testing_struct(1, 2), gen_ulong(2)},
      {gen_testing_overlap(0), gen_testing_overlap(2)}, // overlap
      {gen_testing_struct(1, 2), gen_testing_struct(1, 1)}};
    THEN("Expressions and hashes between pairs should not be equal")
    {
      for(auto &e : expressions)
        test_constructed_differently(e.first, e.second);
    }
  }
}
