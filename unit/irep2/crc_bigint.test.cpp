// H-A4: BigInt CRC ingestion (feed_bigint in irep2_crc.cpp). The magnitude is
// dumped into a 256-byte stack buffer, growing a heap buffer by doubling when
// the value is larger; the sign byte is fed first. This covers the multi-
// doubling heap path (which the existing 512-byte case does not reach) and the
// sign mixing on that path.

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <string>

#include <big-int/bigint.hh>
#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/c_types.h>
#include <util/config.h>

namespace
{
size_t crc_of(const BigInt &v)
{
  return constant_int2tc(get_int_type(64), v)->crc();
}
} // namespace

// ~1500 bytes of magnitude forces several doublings (256->512->1024->2048);
// the resize loop must terminate and the CRC be deterministic and sensitive
// to a byte well past the stack buffer.
TEST_CASE("BigInt CRC over the multi-doubling heap path", "[core][irep2]")
{
  const std::string hex(3000, 'f');
  const BigInt v(hex.c_str(), 16);

  REQUIRE(crc_of(v) == crc_of(v));

  std::string other = hex;
  other.back() = 'e';
  REQUIRE(crc_of(v) != crc_of(BigInt(other.c_str(), 16)));
}

// The sign byte is fed before the magnitude, so a value and its negation
// differ even when the magnitude takes the heap path.
TEST_CASE("BigInt CRC mixes sign on the heap path", "[core][irep2]")
{
  const std::string hex(3000, 'f');
  const BigInt v(hex.c_str(), 16);
  REQUIRE(crc_of(v) != crc_of(-v));
}
