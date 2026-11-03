// SAME-OBJECT over the addresses of two differently named objects folds to
// false. Every shape that may name the same object as something else stays
// for the solver: the same name at another renaming level, a pointer-typed
// or union-typed symbol, the invalid placeholder, and a dereference.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/lang/c_types.h>

namespace
{
type2tc int_ptr()
{
  return pointer_type2tc(get_int32_type());
}

expr2tc var(const type2tc &t, const char *name)
{
  return symbol2tc(t, irep_idt(name));
}

expr2tc addr(const expr2tc &obj)
{
  return address_of2tc(pointer_type2tc(obj->type), obj);
}

expr2tc fold(const expr2tc &a, const expr2tc &b)
{
  expr2tc s = same_object2tc(a, b)->simplify();
  return is_nil_expr(s) ? same_object2tc(a, b) : s;
}
} // namespace

TEST_CASE("&a and &b are different objects", "[irep2][simplify]")
{
  expr2tc a = var(get_int32_type(), "a");
  expr2tc b = var(get_int32_type(), "b");
  REQUIRE(is_false(fold(addr(a), addr(b))));
  REQUIRE(is_true(fold(addr(a), addr(a))));
}

TEST_CASE("an element roots at its array", "[irep2][simplify]")
{
  type2tc arr_t = array_type2tc(get_int32_type(), gen_ulong(4), false);
  expr2tc arr = var(arr_t, "arr");
  expr2tc b = var(get_int32_type(), "b");
  expr2tc elem = index2tc(get_int32_type(), arr, gen_ulong(1));
  REQUIRE(is_false(fold(typecast2tc(int_ptr(), addr(elem)), addr(b))));

  expr2tc first = index2tc(get_int32_type(), arr, gen_ulong(0));
  REQUIRE(is_true(fold(addr(elem), addr(first))));
}

TEST_CASE("one name at two renaming levels is not decided", "[irep2][simplify]")
{
  // A recursive call's local: x at frame 1 and x at frame 2 are two
  // objects, but x at two level-2 versions is one, so neither is assumed.
  expr2tc x1 = symbol2tc(
    get_int32_type(), irep_idt("x"), symbol2t::renaming_level::level1, 1);
  expr2tc x2 = symbol2tc(
    get_int32_type(), irep_idt("x"), symbol2t::renaming_level::level1, 2);
  expr2tc r = fold(addr(x1), addr(x2));
  REQUIRE(!is_constant_bool2t(r));
}

TEST_CASE(
  "pointer-typed and union-typed symbols are not decided",
  "[irep2][simplify]")
{
  expr2tc p = var(int_ptr(), "p");
  expr2tc q = var(int_ptr(), "q");
  REQUIRE(!is_constant_bool2t(fold(addr(p), addr(q))));

  type2tc u_t = union_type2tc(
    std::vector<type2tc>{get_int32_type(), get_uint32_type()},
    std::vector<irep_idt>{"i", "u"},
    std::vector<irep_idt>{"i", "u"},
    "u");
  expr2tc u = var(u_t, "u");
  expr2tc b = var(get_int32_type(), "b");
  REQUIRE(!is_constant_bool2t(fold(addr(u), addr(b))));
}

TEST_CASE("the invalid placeholder is not decided", "[irep2][simplify]")
{
  expr2tc invalid = var(get_int32_type(), "INVALID");
  expr2tc invalid_obj = var(get_int32_type(), "symex::invalid_object3");
  expr2tc b = var(get_int32_type(), "b");
  REQUIRE(!is_constant_bool2t(fold(addr(invalid), addr(b))));
  REQUIRE(!is_constant_bool2t(fold(addr(invalid_obj), addr(b))));
}

TEST_CASE("a dereference is not decided", "[irep2][simplify]")
{
  expr2tc p = var(int_ptr(), "p");
  expr2tc b = var(get_int32_type(), "b");
  expr2tc through_p = addr(dereference2tc(get_int32_type(), p));
  REQUIRE(!is_constant_bool2t(fold(through_p, addr(b))));
  REQUIRE(!is_constant_bool2t(fold(p, addr(b))));
}
