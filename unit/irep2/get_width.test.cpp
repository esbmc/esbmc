// R1: get_width widths for real array/vector/struct types, plus the overflow
// trigger the added 64-bit guards reject.

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <cstdint>
#include <limits>
#include <vector>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/c_types.h>
#include <util/config.h>

namespace
{
expr2tc array_size(unsigned long n)
{
  return constant_int2tc(get_uint_type(64), BigInt(n));
}
} // namespace

TEST_CASE("array/vector/struct get_width for normal sizes", "[core][irep2]")
{
  config.ansi_c.word_size = 64;

  type2tc arr = array_type2tc(get_uint_type(32), array_size(4), false);
  REQUIRE(arr->get_width() == 128);

  type2tc vec = vector_type2tc(get_uint_type(16), array_size(4));
  REQUIRE(vec->get_width() == 64);

  std::vector<type2tc> members{get_int_type(32), get_int_type(8)};
  std::vector<irep_idt> names{"a", "b"};
  type2tc st = struct_type2tc(members, names, names, "s");
  REQUIRE(st->get_width() == 40);
}

TEST_CASE("array get_width overflows unsigned int (R1)", "[core][irep2]")
{
  config.ansi_c.word_size = 64;

  const unsigned long num_elems = 1ul << 29; // * 8-bit subtype => 2^32 bits
  type2tc arr = array_type2tc(get_uint_type(8), array_size(num_elems), false);

  const uint64_t full = (uint64_t)num_elems * 8;
  REQUIRE(full > std::numeric_limits<unsigned int>::max());

  // The guard aborts under asserts-on, so get_width() is only called (and its
  // truncation observed) under NDEBUG.
#ifdef NDEBUG
  REQUIRE(arr->get_width() == static_cast<unsigned int>(full));
#endif
}
