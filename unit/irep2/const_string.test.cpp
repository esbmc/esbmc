// H-A7 — constant_string2t byte reconstruction, verified against the real
// irep2 implementation (constant_string_access in src/irep2/irep2_expr.cpp).
// at(i) combines the w bytes of element i into a value, honouring element
// width (1/2/4 bytes) and endianness, returns 0 for the trailing '\0'
// element and nil past the array end. This suite drives the real
// constant_string2t::at() and checks each of those cases.

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <string>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/c_types.h>
#include <util/config.h>

namespace
{
expr2tc make_string(unsigned elem_bits, unsigned long n, const std::string &b)
{
  type2tc subtype = get_uint_type(elem_bits);
  type2tc arr = array_type2tc(
    subtype, constant_int2tc(get_uint_type(64), BigInt(n)), false);
  return constant_string2tc(arr, irep_idt(b), constant_string_kindt::DEFAULT);
}

uint64_t at_val(const expr2tc &s, size_t i)
{
  return to_constant_int2t(to_constant_string2t(s).at(i)).value.to_uint64();
}
} // namespace

TEST_CASE("constant_string 8-bit element reconstruction", "[core][irep2]")
{
  config.ansi_c.word_size = 32;
  config.ansi_c.endianess = configt::ansi_ct::IS_LITTLE_ENDIAN;

  expr2tc s = make_string(8, 4, "ABC"); // 3 chars + 1 '\0' slot
  REQUIRE(at_val(s, 0) == 'A');
  REQUIRE(at_val(s, 1) == 'B');
  REQUIRE(at_val(s, 2) == 'C');
  REQUIRE(at_val(s, 3) == 0);                          // '\0' element
  REQUIRE(is_nil_expr(to_constant_string2t(s).at(4))); // past the end
}

TEST_CASE("constant_string 16-bit element endianness", "[core][irep2]")
{
  config.ansi_c.word_size = 32;
  const std::string bytes("\x01\x02\x03\x04", 4);

  config.ansi_c.endianess = configt::ansi_ct::IS_LITTLE_ENDIAN;
  expr2tc le = make_string(16, 2, bytes);
  REQUIRE(at_val(le, 0) == 0x0201);
  REQUIRE(at_val(le, 1) == 0x0403);

  config.ansi_c.endianess = configt::ansi_ct::IS_BIG_ENDIAN;
  expr2tc be = make_string(16, 2, bytes);
  REQUIRE(at_val(be, 0) == 0x0102);
  REQUIRE(at_val(be, 1) == 0x0304);
}

TEST_CASE("constant_string 32-bit element reconstruction", "[core][irep2]")
{
  config.ansi_c.word_size = 32;
  const std::string bytes("\x01\x02\x03\x04", 4);

  config.ansi_c.endianess = configt::ansi_ct::IS_LITTLE_ENDIAN;
  REQUIRE(at_val(make_string(32, 1, bytes), 0) == 0x04030201);

  config.ansi_c.endianess = configt::ansi_ct::IS_BIG_ENDIAN;
  REQUIRE(at_val(make_string(32, 1, bytes), 0) == 0x01020304);
}
