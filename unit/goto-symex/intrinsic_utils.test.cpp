/*******************************************************************
 Module: Tests for the intrinsic functions

 Author: Rafael Sá Menezes

 Date: May 2025

 Test Plan:
   - Primitive memcpy
 \*******************************************************************/

#include <cstdint>
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <goto-symex/goto_symex.h>

const mode_table_et mode_table[] = {};

SCENARIO("the memcpy generation can generate valid results", "[symex]")
{
  auto primitive_test_case = [](
                               const uint32_t src_width,
                               const uint32_t src_value,
                               const uint32_t src_offset,
                               const uint32_t dst_width,
                               const uint32_t dst_value,
                               const uint32_t dst_offset,
                               const size_t number_of_bytes,
                               const uint32_t expected_value) {
    const expr2tc src =
      constant_int2tc(get_uint_type(src_width), BigInt(src_value));
    const expr2tc dst =
      constant_int2tc(get_uint_type(dst_width), BigInt(dst_value));

    expr2tc result = goto_symex_utils::gen_byte_memcpy(
      src, dst, number_of_bytes, src_offset, dst_offset);

    simplify(result);

    REQUIRE(to_constant_int2t(result).value.to_uint64() == expected_value);
    return result;
  };

  GIVEN("memcpy(&some-char1, &some-char2, 1)")
  {
    primitive_test_case(8, 0xFA, 0, 8, 0x0F, 0, 1, 0xFA);
  }

  GIVEN("memcpy(&some-int1, &some-int2, 4)")
  {
    primitive_test_case(32, 0x01020304, 0, 32, 0x05060708, 0, 4, 0x01020304);
  }

  GIVEN("memcpy(&some-int1[2], &some-int2, 2) with different src offsets")
  {
    primitive_test_case(32, 0x01020304, 2, 32, 0x05060708, 0, 2, 0x05060102);
  }

  GIVEN("memcpy(&some-int1[2], &some-int2, 2) with different dst offsets")
  {
    primitive_test_case(32, 0x01020304, 0, 32, 0x05060708, 2, 2, 0x03040708);
  }
}
