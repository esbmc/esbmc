// The null-comparison fold decides `&sym == NULL` but must refuse an
// address_of over a dereference: `&*p` is p and may be NULL. Pinned
// here because no C input reaches the fold with that shape — clang
// folds `&*p` to `p` (C99 6.5.3.2p3) and `&p->m` lowers to pointer
// arithmetic over the refused base symbol; the shape is internal.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/lang/c_types.h>
#include <util/config/config.h>

namespace
{
type2tc int_ptr()
{
  return pointer_type2tc(get_int32_type());
}

expr2tc null_ptr()
{
  return symbol2tc(int_ptr(), irep_idt("NULL"));
}
} // namespace

TEST_CASE("&obj == NULL folds to false", "[irep2][simplify]")
{
  expr2tc obj = symbol2tc(get_int32_type(), irep_idt("obj"));
  expr2tc eq = equality2tc(address_of2tc(int_ptr(), obj), null_ptr());
  expr2tc s = eq->simplify();
  REQUIRE(is_constant_bool2t(s));
  REQUIRE(!to_constant_bool2t(s).value);
}

TEST_CASE("&*p == NULL must not fold", "[irep2][simplify]")
{
  expr2tc p = symbol2tc(int_ptr(), irep_idt("p"));
  expr2tc roundtrip =
    address_of2tc(int_ptr(), dereference2tc(get_int32_type(), p));
  expr2tc eq = equality2tc(roundtrip, null_ptr());
  expr2tc s = eq->simplify();
  if (!is_nil_expr(s))
    REQUIRE(!is_constant_bool2t(s));
}

TEST_CASE("(T *)(&obj + k) == NULL folds to false", "[irep2][simplify]")
{
  expr2tc obj = symbol2tc(get_int32_type(), irep_idt("obj"));
  expr2tc arith = add2tc(
    int_ptr(),
    address_of2tc(int_ptr(), obj),
    constant_int2tc(get_int32_type(), BigInt(2)));
  expr2tc eq = equality2tc(typecast2tc(int_ptr(), arith), null_ptr());
  expr2tc s = eq->simplify();
  REQUIRE(is_constant_bool2t(s));
  REQUIRE(!to_constant_bool2t(s).value);
}

TEST_CASE("truncating cast chain must not fold", "[irep2][simplify]")
{
  // (int *)(uint8_t)(uintptr_t)&obj: the 8-bit link can be zero even
  // though &obj is not, so the peel must stop at the integer cast.
  expr2tc obj = symbol2tc(get_int32_type(), irep_idt("obj"));
  expr2tc chain = typecast2tc(
    int_ptr(),
    typecast2tc(
      get_uint8_type(),
      typecast2tc(get_uint64_type(), address_of2tc(int_ptr(), obj))));
  expr2tc eq = equality2tc(chain, null_ptr());
  expr2tc s = eq->simplify();
  if (!is_nil_expr(s))
    REQUIRE(!is_constant_bool2t(s));
}

TEST_CASE("&(*p)[i] + k == NULL must not fold", "[irep2][simplify]")
{
  // The peel must bottom out at the chain root: an index over a
  // dereference roots at p, which may be NULL.
  type2tc arr = array_type2tc(get_int32_type(), expr2tc(), true);
  expr2tc p = symbol2tc(pointer_type2tc(arr), irep_idt("p"));
  expr2tc elem = index2tc(
    get_int32_type(),
    dereference2tc(arr, p),
    constant_int2tc(get_int32_type(), BigInt(1)));
  expr2tc arith = add2tc(
    int_ptr(),
    address_of2tc(int_ptr(), elem),
    constant_int2tc(get_int32_type(), BigInt(2)));
  expr2tc eq = equality2tc(arith, null_ptr());
  expr2tc s = eq->simplify();
  if (!is_nil_expr(s))
    REQUIRE(!is_constant_bool2t(s));
}
