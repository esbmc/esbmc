/*******************************************************************
 Module: Tests for the intrinsic functions

 Author: Rafael Sá Menezes

 Date: May 2025

 Test Plan:
   - Primitive memcpy
 \*******************************************************************/

#include "c_types.h"
#include "fmt/format.h"
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
                               const uint32_t expected_value)
  {
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

SCENARIO("do_memcpy_expression for primitive types", "[symex]")
{
  auto test_primitive_case = [](
                               uint32_t src_value,
                               uint32_t dst_value,
                               size_t number_of_bytes,
                               uint32_t expected_value)
  {
    const expr2tc src = constant_int2tc(get_uint_type(32), BigInt(src_value));
    const expr2tc dst = constant_int2tc(get_uint_type(32), BigInt(dst_value));

    expr2tc result =
      goto_symex_utils::do_memcpy_expression(dst, 0, src, 0, number_of_bytes);

    simplify(result);

    REQUIRE(to_constant_int2t(result).value.to_uint64() == expected_value);

    uint32_t A = src_value, B = dst_value;
    memcpy(&B, &A, number_of_bytes);
    REQUIRE(to_constant_int2t(result).value.to_uint64() == expected_value);
  };

  GIVEN("memcpy(&some-int1, &some-int2, 1)")
  {
    test_primitive_case(0xFA, 0x0F, 1, 0xFA);
  }

  GIVEN("memcpy(&some-int1, &some-int2, 4)")
  {
    test_primitive_case(0x01020304, 0x05060708, 4, 0x01020304);
  }
}

SCENARIO("do_memcpy_expression for dst arrays and src primitives", "[symex]")
{
  const type2tc t = get_int_type(8);
  const type2tc arr_t = array_type2tc(t, constant_int2tc(t, BigInt(8)), false);
  const expr2tc src_primitive =
    constant_int2tc(get_uint_type(32), BigInt(0xdeadbeef));
  const expr2tc dst_array = constant_array2tc(
    arr_t,
    std::vector<expr2tc>{
      constant_int2tc(t, BigInt(0x01)),
      constant_int2tc(t, BigInt(0x02)),
      constant_int2tc(t, BigInt(0x03)),
      constant_int2tc(t, BigInt(0x04)),
      constant_int2tc(t, BigInt(0x05)),
      constant_int2tc(t, BigInt(0x06)),
      constant_int2tc(t, BigInt(0x07)),
      constant_int2tc(t, BigInt(0x08))});

  auto test_case = [](
                     const expr2tc &src,
                     const expr2tc &dst,
                     size_t number_of_bytes,
                     const expr2tc &expected_result,
                     size_t dst_offset = 0,
                     size_t src_offset = 0)
  {
    expr2tc result = goto_symex_utils::do_memcpy_expression(
      dst, dst_offset, src, src_offset, number_of_bytes);

    if (!result)
    {
      REQUIRE(false);
    }

    simplify(result);

    REQUIRE(result == expected_result);
  };

  GIVEN("memcpy(some-array, some-int, 4)")
  {
    const expr2tc expected_array = constant_array2tc(
      arr_t,
      std::vector<expr2tc>{
        constant_int2tc(t, BigInt(0xde)),
        constant_int2tc(t, BigInt(0xad)),
        constant_int2tc(t, BigInt(0xbe)),
        constant_int2tc(t, BigInt(0xef)),
        constant_int2tc(t, BigInt(0x05)),
        constant_int2tc(t, BigInt(0x06)),
        constant_int2tc(t, BigInt(0x07)),
        constant_int2tc(t, BigInt(0x08))});
    test_case(src_primitive, dst_array, 4, expected_array);
  }
#if 0
  GIVEN("memcpy(some-array, some-int, 2)")
  {
    expr2tc expected_array = constant_array2tc(
      arr_t,
      std::vector<expr2tc>{
        constant_int2tc(t, BigInt(0x01)),
        constant_int2tc(t, BigInt(0x02)),
        constant_int2tc(t, BigInt(0x07)),
        constant_int2tc(t, BigInt(0x08))});

    test_case(src_primitive, dst_array, 2, expected_array, 2);
  }
#endif
}

SCENARIO("do_memcpy_expression for arrays", "[symex]")
{
  const type2tc t = get_uint_type(32);
  const type2tc arr_t = array_type2tc(t, constant_int2tc(t, BigInt(4)), false);
  const expr2tc src_array = constant_array2tc(
    arr_t,
    std::vector<expr2tc>{
      constant_int2tc(t, BigInt(0x01)),
      constant_int2tc(t, BigInt(0x02)),
      constant_int2tc(t, BigInt(0x03)),
      constant_int2tc(t, BigInt(0x04))});
  const expr2tc dst_array = constant_array2tc(
    arr_t,
    std::vector<expr2tc>{
      constant_int2tc(t, BigInt(0x05)),
      constant_int2tc(t, BigInt(0x06)),
      constant_int2tc(t, BigInt(0x07)),
      constant_int2tc(t, BigInt(0x08))});

  auto test_array_case = [](
                           const expr2tc &src,
                           const expr2tc &dst,
                           size_t number_of_bytes,
                           const expr2tc &expected_result,
                           size_t src_offset = 0,
                           size_t dst_offset = 0)
  {
    expr2tc result = goto_symex_utils::do_memcpy_expression(
      dst, dst_offset, src, src_offset, number_of_bytes);

    if (!result)
    {
      REQUIRE(false);
    }

    simplify(result);

    REQUIRE(result == expected_result);
  };

  auto test_array_fail = [](
                           const expr2tc &src,
                           const expr2tc &dst,
                           size_t number_of_bytes,
                           size_t src_offset = 0,
                           size_t dst_offset = 0)
  {
    expr2tc result = goto_symex_utils::do_memcpy_expression(
      dst, dst_offset, src, src_offset, number_of_bytes);

    REQUIRE_FALSE(result);
  };

  GIVEN("memcpy(some-array, some-array, misalignement)")
  {
    test_array_fail(src_array, dst_array, 7);
    test_array_fail(src_array, dst_array, 8, 2, 0);
    test_array_fail(src_array, dst_array, 8, 0, 2);
  }

  GIVEN("memcpy(some-array, some-array, array-size)")
  {
    // Should short-circuit!
    expr2tc expected_array = constant_array2tc(
      arr_t,
      std::vector<expr2tc>{
        constant_int2tc(t, BigInt(0x01)),
        constant_int2tc(t, BigInt(0x02)),
        constant_int2tc(t, BigInt(0x03)),
        constant_int2tc(t, BigInt(0x04))});

    test_array_case(src_array, dst_array, 16, expected_array);
  }

  GIVEN("memcpy(some-array, some-array, half-array)")
  {
    expr2tc expected_array = constant_array2tc(
      arr_t,
      std::vector<expr2tc>{
        constant_int2tc(t, BigInt(0x01)),
        constant_int2tc(t, BigInt(0x02)),
        constant_int2tc(t, BigInt(0x07)),
        constant_int2tc(t, BigInt(0x08))});

    test_array_case(src_array, dst_array, 8, expected_array);
  }

  GIVEN("memcpy(some-array, some-array, src offset)")
  {
    expr2tc expected_array = constant_array2tc(
      arr_t,
      std::vector<expr2tc>{
        constant_int2tc(t, BigInt(0x02)),
        constant_int2tc(t, BigInt(0x03)),
        constant_int2tc(t, BigInt(0x07)),
        constant_int2tc(t, BigInt(0x08))});

    test_array_case(src_array, dst_array, 8, expected_array, 4, 0);
  }

  GIVEN("memcpy(some-array, some-array, dst offset)")
  {
    expr2tc expected_array = constant_array2tc(
      arr_t,
      std::vector<expr2tc>{
        constant_int2tc(t, BigInt(0x05)),
        constant_int2tc(t, BigInt(0x01)),
        constant_int2tc(t, BigInt(0x02)),
        constant_int2tc(t, BigInt(0x08))});

    test_array_case(src_array, dst_array, 8, expected_array, 0, 4);
  }

  GIVEN("memcpy(some-array, some-array, both offset)")
  {
    expr2tc expected_array = constant_array2tc(
      arr_t,
      std::vector<expr2tc>{
        constant_int2tc(t, BigInt(0x05)),
        constant_int2tc(t, BigInt(0x02)),
        constant_int2tc(t, BigInt(0x03)),
        constant_int2tc(t, BigInt(0x08))});

    test_array_case(src_array, dst_array, 8, expected_array, 4, 4);
  }
}
#if 0

SCENARIO("do_memcpy_expression for structs", "[symex]")
{
  auto test_struct_case = [](const exprt &src, const exprt &dst, size_t number_of_bytes, const exprt &expected_result)
  {
    expr2tc result = goto_symex_utils::do_memcpy_expression(
      dst, 0, src, 0, number_of_bytes);

    simplify(result);

    REQUIRE(result == expected_result);
  };

  GIVEN("memcpy(&some-struct1.field1, &some-struct2.field1, 4)")
  {
    struct_type t;
    t.add_member(member_symbol("field1", get_uint_type(32), t.id()));
    symbolt sym1(t);
    sym1.name = "struct1";
    symbolt sym2(t);
    sym2.name = "struct2";

    exprt src_struct = constant_struct2tc(t, {constant_int2tc(get_uint_type(32), BigInt(0x01020304))});
    exprt dst_struct = constant_struct2tc(t, {constant_int2tc(get_uint_type(32), BigInt(0x05060708))});

    test_struct_case(src_struct, dst_struct, 4, src_struct);
  }
}

SCENARIO("do_memcpy_expression for unions", "[symex]")
{
  auto test_union_case = [](const exprt &src, const exprt &dst, size_t number_of_bytes, const exprt &expected_result)
  {
    expr2tc result = goto_symex_utils::do_memcpy_expression(
      dst, 0, src, 0, number_of_bytes);

    simplify(result);

    REQUIRE(result == expected_result);
  };

  GIVEN("memcpy(&some-union1.field1, &some-union2.field1, 4)")
  {
    union_type t;
    t.add_member(member_symbol("field1", get_uint_type(32), t.id()));
    symbolt sym1(t);
    sym1.name = "union1";
    symbolt sym2(t);
    sym2.name = "union2";

    exprt src_union = constant_union2tc(t, {constant_int2tc(get_uint_type(32), BigInt(0x01020304))});
    exprt dst_union = constant_union2tc(t, {constant_int2tc(get_uint_type(32), BigInt(0x05060708))});

    test_union_case(src_union, dst_union, 4, src_union);
  }
}
#endif
