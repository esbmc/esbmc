/*******************************************************************
 Module: Tests for the intrinsic functions

 Author: Rafael Sá Menezes

 Date: May 2025

 Test Plan:
   - Primitive memcpy
 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <goto-symex/goto_symex.h>

const mode_table_et mode_table[] = {};

SCENARIO("the memcpy generation can generate valid results", "[symex]")
{
  expr2tc src;
  expr2tc dst;
  expr2tc result;

  uint64_t number_of_bytes, src_offset, dst_offset;

  GIVEN("memcpy(&some-char1, &some-char2, 1)")
  {
    src = constant_int2tc(get_int8_type(), BigInt(0xFA));
    dst = constant_int2tc(get_int8_type(), BigInt(0x0F));
    number_of_bytes = 1;
    src_offset = 0;
    dst_offset = 0;

    result = goto_symex_utils::gen_byte_memcpy(
      dst, src, number_of_bytes, src_offset, dst_offset);
    
    simplify(result);

    REQUIRE(result == dst);
  }
}
